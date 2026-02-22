"""
Pydantic models for type-safe knowledge representation
Ensures data integrity across domain boundaries
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, ConfigDict
import numpy as np
import pandas as pd

class DomainType(str, Enum):
    """Standardized domain classifications"""
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    TEXTUAL = "textual"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    MULTIMODAL = "multimodal"

class DataFormat(str, Enum):
    """Supported data formats across domains"""
    JSON = "json"
    CSV = "csv"
    NUMPY = "numpy"
    PANDAS = "pandas"
    TEXT = "text"
    BINARY = "binary"

class KnowledgePriority(int, Enum):
    """Priority levels for knowledge exchange"""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    BACKGROUND = 1

class DomainSchema(BaseModel):
    """Schema definition for domain knowledge structure"""
    domain_id: str = Field(..., min_length=3, max_length=50)
    domain_type: DomainType
    supported_formats: List[DataFormat]
    required_fields: List[str] = []
    optional_fields: List[str] = []
    validation_rules: Dict[str, Any] = {}
    metadata_schema: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(frozen=True)  # Immutable after creation

class KnowledgePayload(BaseModel):
    """Type-safe container for cross-domain knowledge"""
    source_domain: str = Field(..., min_length=3, max_length=50)
    target_domains: List[str] = []
    payload_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core knowledge data with validation
    data: Union[Dict[str, Any], List[Any], str, bytes]
    data_format: DataFormat
    data_type: str = Field(..., min_length=1)
    
    # Exchange metadata
    priority: KnowledgePriority = KnowledgePriority.MEDIUM
    ttl_seconds: Optional[int] = Field(None, ge=1)  # Time-to-live
    requires_acknowledgment: bool = False
    context_tags: List[str] = []
    
    # Compatibility markers
    harmonized: bool = False
    compatibility_score: float = Field(0.0, ge=0.0, le=1.0)
    
    @validator('data')
    def validate_data_format(cls, v, values):
        """Ensure data matches declared format"""
        if 'data_format' not in values:
            return v
            
        data_format = values['data_format']
        
        if data_format == DataFormat.JSON:
            if not isinstance(v, (dict, list)):
                raise ValueError('JSON format requires dict or list')
        elif data_format == DataFormat.NUMPY:
            if not isinstance(v, np.ndarray):
                raise ValueError('NUMPY format requires numpy array')
        elif data_format == DataFormat.PANDAS:
            if not isinstance(v, (pd.DataFrame, pd.Series)):
                raise ValueError('PANDAS format requires DataFrame or Series')
        elif data_format == DataFormat.CSV:
            if not isinstance(v, str):
                raise ValueError('CSV format requires string')
        elif data_format == DataFormat.BINARY:
            if not isinstance(v, bytes):
                raise ValueError('BINARY format requires bytes')
                
        return v
    
    def to_firestore_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-compatible dictionary"""
        result = self.model_dump()
        
        # Convert special types
        result['timestamp'] = self.timestamp
        if isinstance(self.data, (np.ndarray, pd.DataFrame, pd.Series)):
            # Store metadata for reconstruction
            result['data'] = {
                'type': type(self.data).__name__,
                'shape': self.data.shape if hasattr(self.data, 'shape') else None,
                'dtype': str(self.data.dtype) if hasattr(self.data, 'dtype') else None,
                'pickled': True  # Mark for special handling
            }
        
        return result

class ExchangeContext(BaseModel):
    """Context for knowledge exchange operations"""
    session_id: str
    initiating_domain: str
    domains_involved: List[str]
    exchange_start: datetime = Field(default_factory=datetime.utcnow)
    exchange_end: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = {}
    
    def mark_complete(self, success: bool = True):
        """Mark exchange as complete"""
        self.exchange_end = datetime.utcnow()
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1