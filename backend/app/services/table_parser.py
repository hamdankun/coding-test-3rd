"""
Table parser service for extracting and classifying tables from PDFs
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger(__name__)


class TableParser:
    """Parse and classify tables extracted from PDF documents"""
    
    # Table classification keywords
    CAPITAL_CALLS_KEYWORDS = [
        "capital call", "capital calls", "called", "call date", "call amount",
        "contribution", "capital contribution", "call", "drawdown"
    ]
    
    DISTRIBUTIONS_KEYWORDS = [
        "distribution", "distributions", "distributed", "distribution date",
        "distribution amount", "return", "returns", "payout", "dividend"
    ]
    
    ADJUSTMENTS_KEYWORDS = [
        "adjustment", "adjustments", "recallable", "rebalance", "fee",
        "clawback", "recall", "reversal"
    ]
    
    def __init__(self):
        """Initialize table parser"""
        pass
    
    def parse_table(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """
        Parse a table and extract structured data
        
        Args:
            table_data: Raw table data from pdfplumber
            
        Returns:
            Parsed table with metadata
        """
        if not table_data or len(table_data) < 2:
            return {"error": "Invalid table structure", "raw": table_data}
        
        try:
            # Extract headers (first row)
            headers = self._clean_row(table_data[0])
            
            # Extract data rows
            data_rows = []
            for row in table_data[1:]:
                cleaned_row = self._clean_row(row)
                if cleaned_row and any(cell for cell in cleaned_row):  # Non-empty row
                    data_rows.append(cleaned_row)
            
            return {
                "headers": headers,
                "rows": data_rows,
                "row_count": len(data_rows),
                "column_count": len(headers)
            }
        except Exception as e:
            logger.error(f"Error parsing table: {e}")
            return {"error": str(e), "raw": table_data}
    
    def classify_table(self, table_data: Dict[str, Any]) -> str:
        """
        Classify table type (capital calls, distributions, adjustments, etc.)
        
        Args:
            table_data: Parsed table data
            
        Returns:
            Table classification string
        """
        if "error" in table_data:
            return "unknown"
        
        headers_text = " ".join(table_data.get("headers", [])).lower()
        rows_text = " ".join([
            " ".join(str(cell) for cell in row) 
            for row in table_data.get("rows", [])
        ]).lower()
        
        combined_text = headers_text + " " + rows_text
        
        # Count keyword matches
        capital_calls_score = sum(
            combined_text.count(keyword) 
            for keyword in self.CAPITAL_CALLS_KEYWORDS
        )
        
        distributions_score = sum(
            combined_text.count(keyword) 
            for keyword in self.DISTRIBUTIONS_KEYWORDS
        )
        
        adjustments_score = sum(
            combined_text.count(keyword) 
            for keyword in self.ADJUSTMENTS_KEYWORDS
        )
        
        # Return classification based on highest score
        scores = {
            "capital_calls": capital_calls_score,
            "distributions": distributions_score,
            "adjustments": adjustments_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return "unknown"
        
        return max(scores, key=scores.get)
    
    def extract_transactions(self, table_data: Dict[str, Any], table_type: str = None) -> List[Dict[str, Any]]:
        """
        Extract transaction data from classified tables
        
        Args:
            table_data: Parsed table data
            table_type: Optional table classification
            
        Returns:
            List of extracted transactions
        """
        if "error" in table_data or not table_data.get("rows"):
            return []
        
        # Determine table type if not provided
        if not table_type:
            table_type = self.classify_table(table_data)
        
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        transactions = []
        
        try:
            for row in rows:
                if not any(cell for cell in row):  # Skip empty rows
                    continue
                
                # Create row dict mapping headers to values
                row_dict = dict(zip(headers, row))
                
                # Extract transaction based on type
                if table_type == "capital_calls":
                    transaction = self._extract_capital_call(row_dict, headers, row)
                elif table_type == "distributions":
                    transaction = self._extract_distribution(row_dict, headers, row)
                elif table_type == "adjustments":
                    transaction = self._extract_adjustment(row_dict, headers, row)
                else:
                    continue
                
                if transaction:
                    transactions.append(transaction)
        
        except Exception as e:
            logger.error(f"Error extracting transactions: {e}")
        
        return transactions
    
    def _clean_row(self, row: List[Any]) -> List[str]:
        """Clean and normalize a table row"""
        return [str(cell).strip() if cell else "" for cell in row]
    
    def _extract_capital_call(self, row_dict: Dict, headers: List, row: List) -> Optional[Dict[str, Any]]:
        """Extract capital call transaction"""
        transaction = {
            "type": "capital_call"
        }
        
        # Find date column
        date_value = self._find_and_parse_date(row_dict, headers, row)
        if not date_value:
            return None
        transaction["call_date"] = date_value
        
        # Find amount column
        amount_value = self._find_and_parse_amount(row_dict, headers, row, 
                                                   ["amount", "called", "call amount", "capital"])
        if amount_value is None:
            return None
        transaction["amount"] = amount_value
        
        # Find description
        description = self._find_description(row_dict, headers)
        if description:
            transaction["description"] = description
        
        return transaction
    
    def _extract_distribution(self, row_dict: Dict, headers: List, row: List) -> Optional[Dict[str, Any]]:
        """Extract distribution transaction"""
        transaction = {
            "type": "distribution"
        }
        
        # Find date column
        date_value = self._find_and_parse_date(row_dict, headers, row)
        if not date_value:
            return None
        transaction["distribution_date"] = date_value
        
        # Find amount column
        amount_value = self._find_and_parse_amount(row_dict, headers, row,
                                                   ["amount", "distributed", "distribution amount", "payout"])
        if amount_value is None:
            return None
        transaction["amount"] = amount_value
        
        # Check if recallable
        is_recallable = self._find_boolean(row_dict, headers, ["recallable", "recalled", "recall"])
        transaction["is_recallable"] = is_recallable
        
        # Find description
        description = self._find_description(row_dict, headers)
        if description:
            transaction["description"] = description
        
        return transaction
    
    def _extract_adjustment(self, row_dict: Dict, headers: List, row: List) -> Optional[Dict[str, Any]]:
        """Extract adjustment transaction"""
        transaction = {
            "type": "adjustment"
        }
        
        # Find date column
        date_value = self._find_and_parse_date(row_dict, headers, row)
        if not date_value:
            return None
        transaction["adjustment_date"] = date_value
        
        # Find amount column
        amount_value = self._find_and_parse_amount(row_dict, headers, row,
                                                   ["amount", "adjustment amount", "value"])
        if amount_value is None:
            return None
        transaction["amount"] = amount_value
        
        # Find adjustment type
        adj_type = self._find_adjustment_type(row_dict, headers)
        if adj_type:
            transaction["adjustment_type"] = adj_type
        
        # Check if contribution adjustment
        is_contribution = self._find_boolean(row_dict, headers, ["contribution", "call"])
        transaction["is_contribution_adjustment"] = is_contribution
        
        # Find description
        description = self._find_description(row_dict, headers)
        if description:
            transaction["description"] = description
        
        return transaction
    
    def _find_and_parse_date(self, row_dict: Dict, headers: List, row: List) -> Optional[str]:
        """Find and parse date from row"""
        date_keywords = ["date", "call date", "distribution date", "adjustment date", "when"]
        
        for keyword in date_keywords:
            for header in headers:
                if keyword.lower() in header.lower():
                    date_str = row_dict.get(header, "").strip()
                    if date_str:
                        parsed_date = self._parse_date(date_str)
                        if parsed_date:
                            return parsed_date
        
        # Try parsing from row values
        for cell in row:
            cell_str = str(cell).strip()
            parsed_date = self._parse_date(cell_str)
            if parsed_date:
                return parsed_date
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to ISO format (YYYY-MM-DD)"""
        if not date_str:
            return None
        
        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y/%m/%d"
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    
    def _find_and_parse_amount(self, row_dict: Dict, headers: List, row: List, 
                               keywords: List[str]) -> Optional[float]:
        """Find and parse amount from row"""
        for keyword in keywords:
            for header in headers:
                if keyword.lower() in header.lower():
                    amount_str = row_dict.get(header, "").strip()
                    if amount_str:
                        parsed_amount = self._parse_amount(amount_str)
                        if parsed_amount is not None:
                            return parsed_amount
        
        # Try parsing from row values
        for cell in row:
            cell_str = str(cell).strip()
            parsed_amount = self._parse_amount(cell_str)
            if parsed_amount is not None and parsed_amount != 0:
                return parsed_amount
        
        return None
    
    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse various amount formats to float"""
        if not amount_str:
            return None
        
        # Remove common currency symbols and formatting
        cleaned = amount_str.replace("$", "").replace(",", "").strip()
        
        # Remove parentheses and convert to negative
        if "(" in cleaned and ")" in cleaned:
            cleaned = cleaned.replace("(", "-").replace(")", "")
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _find_description(self, row_dict: Dict, headers: List) -> Optional[str]:
        """Find description from row"""
        desc_keywords = ["description", "note", "notes", "details", "reason", "comment"]
        
        for keyword in desc_keywords:
            for header in headers:
                if keyword.lower() in header.lower():
                    desc = row_dict.get(header, "").strip()
                    if desc and len(desc) > 0:
                        return desc
        
        return None
    
    def _find_boolean(self, row_dict: Dict, headers: List, keywords: List[str]) -> bool:
        """Find boolean flag from row"""
        for keyword in keywords:
            for header in headers:
                if keyword.lower() in header.lower():
                    value = row_dict.get(header, "").strip().lower()
                    if value in ["yes", "true", "y", "1", "recallable", "recalled"]:
                        return True
        
        return False
    
    def _find_adjustment_type(self, row_dict: Dict, headers: List) -> Optional[str]:
        """Find adjustment type from row"""
        type_keywords = ["type", "adjustment type", "category"]
        
        for keyword in type_keywords:
            for header in headers:
                if keyword.lower() in header.lower():
                    adj_type = row_dict.get(header, "").strip()
                    if adj_type:
                        return adj_type
        
        return None