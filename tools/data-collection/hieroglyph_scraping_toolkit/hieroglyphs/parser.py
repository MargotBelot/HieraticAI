"""
Gardiner Expression Parser Module

This module provides advanced parsing functionality for Gardiner expressions,
converting hierarchical text representations into structured node trees for
hieroglyphic text reconstruction.

Author: Margot
Date: September 2024
"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..utils import timing_decorator


@dataclass
class ParsedNode:
    """
    Enhanced node class representing parsed Gardiner expressions.
    
    This class extends basic node functionality to include contextual information,
    metadata for quality assessment, and performance optimization hints.
    
    Attributes:
        kind: Node type ('code', 'hbox', 'vbox')
        children: List of child nodes
        value: Node value (Gardiner code for 'code' nodes)
        context: Layout context for spacing decisions
        metadata: Additional metadata for optimization
        cache_key: Performance optimization cache key
    """
    kind: str
    children: List['ParsedNode'] = field(default_factory=list)
    value: Optional[str] = None
    context: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Generate cache key for performance optimization."""
        if self.kind == 'code' and self.value:
            cache_string = f"{self.kind}:{self.value}:{self.context}"
            self.cache_key = hashlib.md5(cache_string.encode()).hexdigest()
    
    def __repr__(self):
        """String representation for debugging."""
        if self.kind == 'code':
            return f"ParsedNode({self.kind}:{self.value})"
        else:
            return f"ParsedNode({self.kind}:[{len(self.children)} children])"


class GardinerExpressionParser:
    """
    Advanced parser for Gardiner expressions with enhanced error handling and context awareness.
    
    This parser supports:
    - Complex nested expressions with multiple operators
    - Context-aware parsing for different text types
    - Robust error handling and recovery
    - Performance optimization through intelligent tokenization and caching
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Gardiner expression parser.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("hieroglyph_scraping_toolkit")
        self.parse_cache = {}
        self.error_patterns = []
        self.parsing_statistics = {
            'total_parsed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    @timing_decorator
    def parse_expression(self, expression: str, line_number: int = 0, 
                        context: str = "document") -> ParsedNode:
        """
        Parse a Gardiner expression into an enhanced node tree.
        
        This method handles the complete parsing pipeline including preprocessing,
        tokenization, tree construction, and error recovery.
        
        Args:
            expression: Gardiner expression string to parse
            line_number: Line number for error reporting
            context: Parsing context for optimization hints
            
        Returns:
            ParsedNode: Root node of the parsed expression tree
        """
        self.parsing_statistics['total_parsed'] += 1
        
        # Check cache first
        cache_key = hashlib.md5(f"{expression}:{context}".encode()).hexdigest()
        if cache_key in self.parse_cache:
            self.parsing_statistics['cache_hits'] += 1
            return self.parse_cache[cache_key]
        
        self.parsing_statistics['cache_misses'] += 1
        
        try:
            # Preprocess the expression
            cleaned_expr = self._preprocess_expression(expression)
            
            # Tokenize with context awareness
            tokens = self._tokenize_expression(cleaned_expr, context)
            
            # Parse tokens into tree structure
            root_node = self._parse_tokens(tokens, line_number, context)
            
            # Cache successful parse
            self.parse_cache[cache_key] = root_node
            self.parsing_statistics['successful_parses'] += 1
            
            self.logger.debug(f"Successfully parsed expression: {expression[:50]}...")
            return root_node
            
        except Exception as e:
            self.parsing_statistics['failed_parses'] += 1
            self.logger.error(f"Failed to parse expression '{expression}' on line {line_number}: {e}")
            
            # Record error pattern for analysis
            self.error_patterns.append({
                'expression': expression,
                'line_number': line_number,
                'error': str(e),
                'context': context
            })
            
            # Return error recovery node
            return self._create_error_recovery_node(expression, str(e))
    
    def _preprocess_expression(self, expression: str) -> str:
        """
        Preprocess expression to normalize format and handle edge cases.
        
        This method standardizes the input format by:
        - Removing extra whitespace
        - Normalizing quotation marks
        - Handling common encoding issues
        - Fixing typical formatting problems
        
        Args:
            expression: Raw expression string
            
        Returns:
            str: Preprocessed expression string
        """
        # Remove extra whitespace but preserve structure
        cleaned = re.sub(r'\s+', '', expression)
        
        # Normalize quotation marks to standard single quotes
        quote_replacements = [
            ('"', "'"), ('"', "'"), ('"', "'"),  # Smart quotes
            ('´', "'"), ('`', "'")  # Accent marks used as quotes
        ]
        
        for old_quote, new_quote in quote_replacements:
            cleaned = cleaned.replace(old_quote, new_quote)
        
        # Handle common encoding issues
        encoding_fixes = [
            ('–', '-'), ('—', '-'),  # En dash and em dash to hyphen
            ('…', '...'),  # Ellipsis
            ('′', "'"), ('″', '"')  # Prime symbols
        ]
        
        for old_char, new_char in encoding_fixes:
            cleaned = cleaned.replace(old_char, new_char)
        
        # Remove zero-width characters that might cause issues
        cleaned = re.sub(r'[\u200b-\u200d\ufeff]', '', cleaned)
        
        return cleaned
    
    def _tokenize_expression(self, expression: str, context: str) -> List[str]:
        """
        Advanced tokenization with context awareness and error handling.
        
        This method breaks down the expression into meaningful tokens while
        handling edge cases and providing context-specific optimizations.
        
        Args:
            expression: Preprocessed expression string
            context: Parsing context for optimization
            
        Returns:
            List[str]: List of tokens
        """
        # Enhanced tokenization pattern that handles:
        # - Parentheses and operators
        # - Quoted strings (single quotes)
        # - Special characters and manual placement markers
        # - Gardiner codes with various formats
        token_pattern = r"([:*()]|'[^']*'|[&][^:*()]*|[^:*()]+)"
        
        try:
            raw_tokens = re.findall(token_pattern, expression)
        except re.error as e:
            self.logger.error(f"Regex error in tokenization: {e}")
            # Fallback to simple split
            raw_tokens = re.split(r'([:()*])', expression)
        
        # Clean and validate tokens
        tokens = []
        for token in raw_tokens:
            token = token.strip()
            if token:  # Only add non-empty tokens
                tokens.append(token)
        
        # Context-specific token processing
        if context == "line":
            tokens = self._optimize_line_tokens(tokens)
        elif context == "word":
            tokens = self._optimize_word_tokens(tokens)
        
        return tokens
    
    def _optimize_line_tokens(self, tokens: List[str]) -> List[str]:
        """
        Optimize tokens for line-level parsing context.
        
        Args:
            tokens: List of raw tokens
            
        Returns:
            List[str]: Optimized tokens for line context
        """
        # For line context, we might want to group certain patterns
        # or handle line-specific formatting
        return tokens
    
    def _optimize_word_tokens(self, tokens: List[str]) -> List[str]:
        """
        Optimize tokens for word-level parsing context.
        
        Args:
            tokens: List of raw tokens
            
        Returns:
            List[str]: Optimized tokens for word context
        """
        # For word context, we might want to handle spacing differently
        return tokens
    
    def _parse_tokens(self, tokens: List[str], line_number: int, 
                     context: str) -> ParsedNode:
        """
        Parse tokenized expression into a structured node tree.
        
        This method implements a stack-based parsing algorithm that handles
        nested expressions, multiple operators, and maintains proper precedence.
        
        Args:
            tokens: List of tokens to parse
            line_number: Line number for error reporting
            context: Parsing context
            
        Returns:
            ParsedNode: Root node of the parsed tree
        """
        if not tokens:
            raise ValueError("Empty token list")
        
        # Stack-based parsing with enhanced error handling
        node_stack = [[]]  # Stack of node lists
        operator_stack = [[]]  # Stack of operator lists
        paren_count = 0
        
        for i, token in enumerate(tokens):
            try:
                if token == '(':
                    paren_count += 1
                    node_stack.append([])
                    operator_stack.append([])
                    
                elif token == ')':
                    if paren_count <= 0:
                        raise ValueError(f"Unmatched closing parenthesis at token {i}")
                    
                    paren_count -= 1
                    sub_nodes = node_stack.pop()
                    sub_operators = operator_stack.pop()
                    
                    if sub_nodes:
                        combined_node = self._build_tree_from_components(
                            sub_nodes, sub_operators, context
                        )
                        node_stack[-1].append(combined_node)
                    
                elif token in ['*', ':']:
                    operator_stack[-1].append(token)
                    
                else:
                    # Process glyph code or special token
                    code_node = self._process_glyph_token(token, line_number, context)
                    node_stack[-1].append(code_node)
                    
            except Exception as e:
                self.logger.warning(f"Error processing token '{token}' at position {i}: {e}")
                # Continue with error recovery - skip problematic token
                continue
        
        # Validate final state
        if paren_count != 0:
            raise ValueError(f"Mismatched parentheses: {paren_count} unclosed")
        
        if not node_stack or not node_stack[0]:
            raise ValueError("No valid nodes found after parsing")
        
        # Build final tree
        return self._build_tree_from_components(node_stack[0], operator_stack[0], context)
    
    def _process_glyph_token(self, token: str, line_number: int, context: str) -> ParsedNode:
        """
        Process individual glyph token with comprehensive error handling.
        
        This method handles various token types including:
        - Regular Gardiner codes
        - Manual placement markers (tokens with '&')
        - Special formatting tokens
        - Quoted tokens
        
        Args:
            token: Token to process
            line_number: Line number for error reporting
            context: Parsing context
            
        Returns:
            ParsedNode: Processed node
        """
        # Handle manual placement markers
        if '&' in token:
            parts = token.split('&', 1)
            if parts[0]:
                # Token before '&' is a Gardiner code
                code = parts[0].strip("'").rstrip('-')
                manual_text = parts[1].strip()
                
                self.logger.info(f"Manual placement needed for '{manual_text}' on line {line_number}")
                
                return ParsedNode(
                    kind='code',
                    value=code,
                    context=context,
                    metadata={
                        'manual_placement': manual_text,
                        'line_number': line_number,
                        'requires_attention': True
                    }
                )
            else:
                # Token starts with '&', treat as special marker
                return ParsedNode(
                    kind='code',
                    value="",
                    context=context,
                    metadata={
                        'special_marker': token,
                        'line_number': line_number,
                        'requires_attention': True
                    }
                )
        
        # Handle quoted tokens
        if token.startswith("'") and token.endswith("'"):
            code = token[1:-1].rstrip('-')
        else:
            code = token.rstrip('-')
        
        # Validate Gardiner code format (basic validation)
        if code and self._is_valid_gardiner_code(code):
            return ParsedNode(
                kind='code',
                value=code,
                context=context,
                metadata={'line_number': line_number}
            )
        else:
            # Handle invalid or empty codes
            self.logger.warning(f"Invalid or empty Gardiner code: '{token}' on line {line_number}")
            return ParsedNode(
                kind='code',
                value=code,
                context=context,
                metadata={
                    'line_number': line_number,
                    'invalid_code': True,
                    'original_token': token
                }
            )
    
    def _is_valid_gardiner_code(self, code: str) -> bool:
        """
        Validate if a string appears to be a valid Gardiner code.
        
        This performs basic format validation - a more comprehensive
        validation would require a complete Gardiner sign list.
        
        Args:
            code: Code string to validate
            
        Returns:
            bool: True if code appears to be valid
        """
        if not code:
            return False
        
        # Basic pattern: Letter(s) followed by number(s)
        # Examples: A1, AA12, Aa1, etc.
        pattern = r'^[A-Za-z]{1,3}[0-9]{1,3}[a-z]?$'
        return bool(re.match(pattern, code))
    
    def _build_tree_from_components(self, nodes: List[ParsedNode], 
                                  operators: List[str], context: str) -> ParsedNode:
        """
        Build tree structure from nodes and operators with enhanced logic.
        
        This method constructs a properly structured tree from parsed components,
        handling operator precedence and grouping correctly.
        
        Args:
            nodes: List of parsed nodes
            operators: List of operators between nodes
            context: Parsing context
            
        Returns:
            ParsedNode: Root node of the constructed tree
        """
        if not nodes:
            raise ValueError("Cannot build tree from empty node list")
        
        if len(nodes) == 1:
            return nodes[0]
        
        # Ensure we have the right number of operators
        expected_operators = len(nodes) - 1
        if len(operators) != expected_operators:
            self.logger.warning(
                f"Operator count mismatch: {len(operators)} ops for {len(nodes)} nodes"
            )
            # Pad with default operators if needed
            while len(operators) < expected_operators:
                operators.append('*')  # Default to horizontal layout
            # Trim excess operators
            operators = operators[:expected_operators]
        
        # Build left-associative tree
        result = nodes[0]
        for i, operator in enumerate(operators):
            if i + 1 < len(nodes):
                container_type = 'hbox' if operator == '*' else 'vbox'
                result = ParsedNode(
                    kind=container_type,
                    children=[result, nodes[i + 1]],
                    context=context,
                    metadata={
                        'operator': operator,
                        'tree_level': i + 1
                    }
                )
        
        return result
    
    def _create_error_recovery_node(self, expression: str, error_msg: str) -> ParsedNode:
        """
        Create a recovery node when parsing fails completely.
        
        This method attempts to extract any recognizable content from
        a failed expression to provide partial functionality.
        
        Args:
            expression: Original expression that failed to parse
            error_msg: Error message from the parsing failure
            
        Returns:
            ParsedNode: Error recovery node with extracted content
        """
        self.logger.error(f"Creating error recovery node for: '{expression}' - {error_msg}")
        
        # Try to extract any recognizable Gardiner codes
        gardiner_pattern = r'[A-Z]+[0-9]+'
        codes = re.findall(gardiner_pattern, expression)
        
        if codes:
            # Create nodes for recognized codes
            code_nodes = [
                ParsedNode(
                    kind='code',
                    value=code,
                    context="error_recovery",
                    metadata={
                        'recovered_from_error': True,
                        'original_expression': expression,
                        'error_message': error_msg
                    }
                )
                for code in codes
            ]
            
            if len(code_nodes) == 1:
                return code_nodes[0]
            else:
                return ParsedNode(
                    kind='hbox',
                    children=code_nodes,
                    context="error_recovery",
                    metadata={
                        'recovered_from_error': True,
                        'original_expression': expression,
                        'error_message': error_msg,
                        'recovered_codes': len(code_nodes)
                    }
                )
        else:
            # Last resort: create a placeholder node
            return ParsedNode(
                kind='code',
                value="",
                context="error_recovery",
                metadata={
                    'complete_failure': True,
                    'original_expression': expression,
                    'error_message': error_msg
                }
            )
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive parsing statistics.
        
        Returns:
            Dict[str, Any]: Statistics about parsing performance and errors
        """
        cache_rate = (
            self.parsing_statistics['cache_hits'] / 
            max(1, self.parsing_statistics['cache_hits'] + self.parsing_statistics['cache_misses'])
            * 100
        )
        
        success_rate = (
            self.parsing_statistics['successful_parses'] / 
            max(1, self.parsing_statistics['total_parsed'])
            * 100
        )
        
        return {
            **self.parsing_statistics,
            'cache_rate_percent': cache_rate,
            'success_rate_percent': success_rate,
            'cache_size': len(self.parse_cache),
            'error_patterns_recorded': len(self.error_patterns),
            'recent_errors': self.error_patterns[-5:] if self.error_patterns else []
        }
    
    def clear_cache(self) -> None:
        """Clear the parsing cache and reset statistics."""
        self.parse_cache.clear()
        self.parsing_statistics = {
            'total_parsed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.logger.info("Parser cache and statistics cleared")
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """
        Analyze recorded error patterns to identify common issues.
        
        Returns:
            Dict[str, Any]: Analysis of error patterns
        """
        if not self.error_patterns:
            return {'total_errors': 0, 'common_patterns': []}
        
        # Group errors by type
        error_types = {}
        for error in self.error_patterns:
            error_type = error['error'].split(':')[0]  # Get error type
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        # Find most common error types
        common_patterns = [
            {
                'error_type': error_type,
                'count': len(errors),
                'examples': errors[:3]  # Show first 3 examples
            }
            for error_type, errors in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True)
        ]
        
        return {
            'total_errors': len(self.error_patterns),
            'unique_error_types': len(error_types),
            'common_patterns': common_patterns[:10],  # Top 10 most common
            'error_rate': len(self.error_patterns) / max(1, self.parsing_statistics['total_parsed'])
        }