"""
Query engine service for RAG-based question answering

Implements Retrieval Augmented Generation (RAG) with:
- Intent classification
- Vector similarity search
- Metrics calculation
- LLM-powered response generation
"""
from typing import Dict, Any, List, Optional
import time
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from app.models.fund import Fund
from app.models.transaction import CapitalCall, Distribution, Adjustment
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class QueryEngine:
    """RAG-based query engine for fund analysis"""
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.metrics_calculator = MetricsCalculator(db)
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on available API keys"""
        if settings.OPENAI_API_KEY:
            logger.info("Using OpenAI LLM")
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0,
                openai_api_key=settings.OPENAI_API_KEY
            )
        else:
            # Fallback to local LLM
            logger.info("Using Ollama LLM (ensure Ollama is running)")
            return Ollama(model="llama2")
    
    async def process_query(
        self, 
        query: str, 
        fund_id: Optional[int] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using RAG
        
        Args:
            query: User question
            fund_id: Optional fund ID for context
            conversation_history: Previous conversation messages
            
        Returns:
            Response with answer, sources, and metrics
        """
        start_time = time.time()
        
        try:
            # Step 1: Classify query intent
            intent = self._classify_intent(query)
            logger.info(f"Query intent: {intent}")
            
            # Step 2: Retrieve relevant context from vector store
            filter_metadata = {"fund_id": fund_id} if fund_id else None
            try:
                relevant_docs = await self.vector_store.similarity_search(
                    query=query,
                    k=settings.TOP_K_RESULTS,
                    filter_metadata=filter_metadata
                )
            except Exception as e:
                logger.warning(f"Error retrieving documents from vector store: {e}. Using fallback.")
                relevant_docs = []
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            
            # Step 2b: Fetch structured data for retrieval queries
            sql_data = None
            if intent == "retrieval":
                sql_data = self._fetch_transaction_data(query, fund_id)
                logger.info(f"Retrieved SQL data: {sql_data is not None}")
            
            # Step 3: Calculate metrics if needed
            metrics = None
            if intent == "calculation":
                if fund_id:
                    # Calculate metrics for specific fund
                    metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
                    logger.info(f"Calculated metrics for fund {fund_id}: {list(metrics.keys())}")
                else:
                    # Calculate metrics for all funds when no specific fund is selected
                    all_funds = self.db.query(Fund).all()
                    if all_funds:
                        if len(all_funds) == 1:
                            # If only one fund, return metrics directly without wrapping
                            metrics = self.metrics_calculator.calculate_all_metrics(all_funds[0].id)
                            logger.info(f"Calculated metrics for single fund {all_funds[0].name}: {list(metrics.keys())}")
                        else:
                            # If multiple funds, return as dict with fund names as keys
                            metrics = {}
                            for fund in all_funds:
                                fund_metrics = self.metrics_calculator.calculate_all_metrics(fund.id)
                                metrics[fund.name] = fund_metrics
                            logger.info(f"Calculated metrics for {len(all_funds)} funds: {list(metrics.keys())}")
                    else:
                        logger.warning("No funds found in database")
            
            # Step 3b: Also provide metrics for general queries about performance/comparison
            # This helps LLM give context-aware answers
            if intent == "general" and metrics is None:
                query_lower = query.lower()
                if any(keyword in query_lower for keyword in ["perform", "compari", "benchmark", "outperform", "underperform", "compare"]):
                    if fund_id:
                        metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
                        logger.info(f"Provided metrics context for general query on fund {fund_id}")
                    else:
                        # Get metrics for all funds for comparison queries
                        all_funds = self.db.query(Fund).all()
                        if all_funds:
                            if len(all_funds) == 1:
                                metrics = self.metrics_calculator.calculate_all_metrics(all_funds[0].id)
                            else:
                                metrics = {}
                                for fund in all_funds:
                                    fund_metrics = self.metrics_calculator.calculate_all_metrics(fund.id)
                                    metrics[fund.name] = fund_metrics
                            logger.info(f"Provided metrics context for comparison query")
            
            # Step 4: Generate response using LLM
            answer = await self._generate_response(
                query=query,
                intent=intent,
                context=relevant_docs,
                metrics=metrics,
                sql_data=sql_data,
                conversation_history=conversation_history or []
            )
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc["content"],
                        "metadata": {
                            k: v for k, v in doc.items() 
                            if k not in ["content", "score"]
                        },
                        "score": doc.get("score")
                    }
                    for doc in relevant_docs
                ],
                "metrics": metrics,
                "intent": intent,
                "processing_time": round(processing_time, 2)
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "sources": [],
                "metrics": None,
                "intent": "error",
                "processing_time": round(time.time() - start_time, 2)
            }
    
    def _classify_intent(self, query: str) -> str:
        """
        Classify query intent
        
        Returns:
            'calculation', 'definition', 'retrieval', or 'general'
        """
        query_lower = query.lower()
        
        # Calculation keywords - be more specific to avoid false positives
        calc_keywords = [
            "calculate", "what is the", "current dpi", "current irr", "current tvpi", 
            "what is dpi", "what is irr", "what is tvpi",
            "compute", "figure out", "how much", "total distributions",
            "paid-in capital", "pic", "what's the dpi", "what's the irr"
        ]
        if any(keyword in query_lower for keyword in calc_keywords):
            return "calculation"
        
        # Retrieval keywords (check BEFORE definition keywords)
        # These include trend analysis, data retrieval, and time-series queries
        ret_keywords = [
            "show me", "list", "all", "find", "search", "when", 
            "how many", "which", "get", "retrieve", "trend", "over time",
            "capital call", "distribution", "adjustment", "transaction",
            "timeline", "chronolog", "history", "pattern"
        ]
        if any(keyword in query_lower for keyword in ret_keywords):
            return "retrieval"
        
        # Definition keywords (check AFTER retrieval to avoid false positives)
        def_keywords = [
            "what does", "mean", "define", "explain", "definition", 
            "what is a", "what are", "describe"
        ]
        if any(keyword in query_lower for keyword in def_keywords):
            return "definition"
        
        return "general"
    
    def _fetch_transaction_data(self, query: str, fund_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """
        Fetch structured transaction data from database for retrieval queries
        
        Args:
            query: User query to determine what data to fetch
            fund_id: Optional fund ID to filter by
            
        Returns:
            Dictionary with capital calls, distributions, and adjustments data
        """
        query_lower = query.lower()
        data = {}
        
        try:
            # Determine which fund(s) to query
            if fund_id:
                funds_to_query = self.db.query(Fund).filter(Fund.id == fund_id).all()
            else:
                funds_to_query = self.db.query(Fund).all()
            
            if not funds_to_query:
                return None
            
            fund_ids = [f.id for f in funds_to_query]
            
            # Extract year if mentioned in query
            year_match = None
            for i in range(2020, 2030):
                if str(i) in query:
                    year_match = i
                    break
            
            # Check if asking for trends/timeline (don't filter by year, get all data)
            is_trend_query = any(keyword in query_lower for keyword in ["trend", "over time", "timeline", "chronolog", "history", "pattern"])
            
            # Fetch Capital Calls
            if any(keyword in query_lower for keyword in ["capital call", "calls", "show me all", "trend", "timeline"]):
                capital_calls = self.db.query(CapitalCall).filter(
                    CapitalCall.fund_id.in_(fund_ids)
                ).order_by(CapitalCall.call_date).all()
                
                # Only filter by year if it's NOT a trend query
                if year_match and not is_trend_query:
                    capital_calls = [
                        cc for cc in capital_calls 
                        if cc.call_date.year == year_match
                    ]
                
                if capital_calls:
                    data["capital_calls"] = {
                        "count": len(capital_calls),
                        "total_amount": float(sum(cc.amount for cc in capital_calls)),
                        "date_range": {
                            "start": capital_calls[0].call_date.isoformat(),
                            "end": capital_calls[-1].call_date.isoformat()
                        } if capital_calls else None,
                        "items": [
                            {
                                "date": cc.call_date.isoformat(),
                                "amount": float(cc.amount),
                                "type": cc.call_type or "Standard",
                                "description": cc.description or ""
                            }
                            for cc in capital_calls
                        ]
                    }
            
            # Fetch Distributions
            if any(keyword in query_lower for keyword in ["distribution", "distributions", "paid", "returned", "show me all", "trend", "timeline"]):
                distributions = self.db.query(Distribution).filter(
                    Distribution.fund_id.in_(fund_ids)
                ).order_by(Distribution.distribution_date).all()
                
                # Only filter by year if it's NOT a trend query
                if year_match and not is_trend_query:
                    distributions = [
                        d for d in distributions 
                        if d.distribution_date.year == year_match
                    ]
                
                if distributions:
                    data["distributions"] = {
                        "count": len(distributions),
                        "total_amount": float(sum(d.amount for d in distributions)),
                        "recallable_amount": float(sum(d.amount for d in distributions if d.is_recallable)),
                        "date_range": {
                            "start": distributions[0].distribution_date.isoformat(),
                            "end": distributions[-1].distribution_date.isoformat()
                        } if distributions else None,
                        "items": [
                            {
                                "date": d.distribution_date.isoformat(),
                                "amount": float(d.amount),
                                "type": d.distribution_type or "Standard",
                                "is_recallable": d.is_recallable,
                                "description": d.description or ""
                            }
                            for d in distributions
                        ]
                    }
            
            # Fetch Adjustments
            if any(keyword in query_lower for keyword in ["adjustment", "adjustments", "rebalance", "refund", "trend", "timeline"]):
                adjustments = self.db.query(Adjustment).filter(
                    Adjustment.fund_id.in_(fund_ids)
                ).order_by(Adjustment.adjustment_date).all()
                
                # Only filter by year if it's NOT a trend query
                if year_match and not is_trend_query:
                    adjustments = [
                        a for a in adjustments 
                        if a.adjustment_date.year == year_match
                    ]
                
                if adjustments:
                    data["adjustments"] = {
                        "count": len(adjustments),
                        "total_amount": float(sum(a.amount for a in adjustments)),
                        "date_range": {
                            "start": adjustments[0].adjustment_date.isoformat(),
                            "end": adjustments[-1].adjustment_date.isoformat()
                        } if adjustments else None,
                        "items": [
                            {
                                "date": a.adjustment_date.isoformat(),
                                "amount": float(a.amount),
                                "type": a.adjustment_type or "Standard",
                                "category": a.category or "Other",
                                "description": a.description or ""
                            }
                            for a in adjustments
                        ]
                    }
            
            return data if data else None
        
        except Exception as e:
            logger.error(f"Error fetching transaction data: {e}")
            return None
    
    async def _generate_response(
        self,
        query: str,
        intent: str,
        context: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        sql_data: Optional[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate response using LLM"""
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}]\n{doc['content']}"
            for i, doc in enumerate(context[:3])  # Use top 3 sources
        ])
        
        if not context_str:
            context_str = "No documents available."
        
        # Build metrics string
        metrics_str = ""
        if metrics:
            # Detect if this is multi-fund format (all values are dicts) or single-fund format
            is_multi_fund = isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()) and any(isinstance(v, dict) for v in metrics.values())
            
            if is_multi_fund:
                # Multi-fund format: {"Fund Name": {metric_data}, ...}
                metrics_str = "\n\n**Available Fund Metrics:**\n"
                for fund_name, fund_metrics in metrics.items():
                    if isinstance(fund_metrics, dict):
                        metrics_str += f"\n**{fund_name}:**\n"
                        for key, value in fund_metrics.items():
                            if value is not None:
                                if isinstance(value, float):
                                    metrics_str += f"  - {key.upper()}: {value:.2f}\n"
                                else:
                                    metrics_str += f"  - {key.upper()}: {value}\n"
            else:
                # Single fund format: {metric_key: value, ...}
                metrics_str = "\n\n**Available Fund Metrics:**\n"
                for key, value in metrics.items():
                    if value is not None:
                        if isinstance(value, float):
                            metrics_str += f"- **{key.upper()}**: {value:.2f}\n"
                        else:
                            metrics_str += f"- **{key.upper()}**: {value}\n"
        
        # Build SQL data string for retrieval queries
        sql_data_str = ""
        if sql_data:
            sql_data_str = "\n\n**Transaction Data:**\n"
            
            if "capital_calls" in sql_data:
                cc_data = sql_data["capital_calls"]
                sql_data_str += f"\n**Capital Calls:**\n"
                sql_data_str += f"- Total Count: {cc_data['count']}\n"
                sql_data_str += f"- Total Amount: ${cc_data['total_amount']:,.2f}\n"
                if cc_data.get("date_range"):
                    sql_data_str += f"- Date Range: {cc_data['date_range']['start']} to {cc_data['date_range']['end']}\n"
                if cc_data.get("items"):
                    sql_data_str += "\nDetailed Capital Calls (in chronological order):\n"
                    for item in cc_data["items"]:
                        sql_data_str += f"  - {item['date']}: ${item['amount']:,.2f} ({item['type']})\n"
            
            if "distributions" in sql_data:
                dist_data = sql_data["distributions"]
                sql_data_str += f"\n**Distributions:**\n"
                sql_data_str += f"- Total Count: {dist_data['count']}\n"
                sql_data_str += f"- Total Amount: ${dist_data['total_amount']:,.2f}\n"
                sql_data_str += f"- Recallable Amount: ${dist_data['recallable_amount']:,.2f}\n"
                if dist_data.get("date_range"):
                    sql_data_str += f"- Date Range: {dist_data['date_range']['start']} to {dist_data['date_range']['end']}\n"
                if dist_data.get("items"):
                    sql_data_str += "\nDetailed Distributions (in chronological order):\n"
                    for item in dist_data["items"][:15]:  # Increased limit for trend analysis
                        recallable_str = " (Recallable)" if item['is_recallable'] else ""
                        sql_data_str += f"  - {item['date']}: ${item['amount']:,.2f} ({item['type']}){recallable_str}\n"
            
            if "adjustments" in sql_data:
                adj_data = sql_data["adjustments"]
                sql_data_str += f"\n**Adjustments:**\n"
                sql_data_str += f"- Total Count: {adj_data['count']}\n"
                sql_data_str += f"- Total Amount: ${adj_data['total_amount']:,.2f}\n"
                if adj_data.get("date_range"):
                    sql_data_str += f"- Date Range: {adj_data['date_range']['start']} to {adj_data['date_range']['end']}\n"
                if adj_data.get("items"):
                    sql_data_str += "\nDetailed Adjustments (in chronological order):\n"
                    for item in adj_data["items"][:15]:  # Increased limit for trend analysis
                        sql_data_str += f"  - {item['date']}: ${item['amount']:,.2f} ({item['category']})\n"
        
        # Build conversation history string
        history_str = ""
        if conversation_history:
            history_str = "\n\n**Previous Conversation:**\n"
            for msg in conversation_history[-2:]:  # Last 2 messages
                role = msg.get('role', 'user').capitalize()
                history_str += f"{role}: {msg.get('content', '')}\n"
        
        # Create system prompt based on intent
        system_prompts = {
            "calculation": """You are a financial analyst assistant specializing in private equity fund performance.

When answering calculation questions:
- Use the provided metrics data
- Show your work step-by-step
- Explain any assumptions made
- Format numbers with commas and $ signs when appropriate
- Round to 2 decimal places for percentages""",
            
            "definition": """You are a financial education assistant.

When answering definition questions:
- Use simple, clear language
- Provide real-world examples when possible
- Avoid unnecessary jargon
- Explain related concepts if helpful""",
            
            "retrieval": """You are a data analyst assistant.

When answering retrieval questions:
- Be specific and list exact values
- Use bullet points or tables for multiple items
- Include dates when relevant
- Sort chronologically if asking for time-series data""",
            
            "general": """You are a helpful financial analysis assistant.

Answer questions thoughtfully and provide context when useful."""
        }
        
        system_prompt = system_prompts.get(intent, system_prompts["general"])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{system_prompt}

Always:
- Be concise but thorough
- Use **bold** for important numbers
- Cite your sources from the provided documents
- If you don't have enough information, say so
- Format monetary amounts with $ and percentages with %"""),
            ("user", """Context from documents:
{context}
{metrics}
{sql_data}
{history}

Question: {query}

Please provide a helpful answer based on the context, data, and metrics provided.""")
        ])
        
        # Generate response
        try:
            messages = prompt.format_messages(
                context=context_str,
                metrics=metrics_str,
                sql_data=sql_data_str,
                history=history_str,
                query=query
            )
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def add_to_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        role: str,
        content: str
    ) -> List[Dict[str, str]]:
        """Add a message to conversation history"""
        conversation_history.append({
            "role": role,
            "content": content
        })
        return conversation_history
