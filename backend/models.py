from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class UserProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    telegram_id: Optional[int] = None
    telegram_username: Optional[str] = None
    name: str
    age: int
    gender: str  # "male", "female", "other"
    bio: Optional[str] = None
    photo_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class RentPreferences(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    metro_stations: List[str] = []  # Preferred metro stations
    radius_km: float = 5.0  # Search radius in km
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    rooms_count: Optional[List[int]] = None  # [1, 2, 3] for 1-3 room apartments
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class Coordinates(BaseModel):
    latitude: float
    longitude: float

class Apartment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    price: Optional[int] = None
    address: str
    coordinates: Optional[Coordinates] = None
    metro_station: Optional[str] = None
    rooms_count: Optional[int] = None
    area: Optional[float] = None  # Area in square meters
    floor: Optional[int] = None
    total_floors: Optional[int] = None
    photos: List[str] = []  # URLs to photos
    contact_info: Optional[str] = None  # Phone, telegram, etc.
    telegram_channel: Optional[str] = None  # Source channel
    message_id: Optional[int] = None  # Original message ID
    parsed_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True

class UserLike(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_user_id: str
    to_user_id: str
    created_at: datetime = Field(default_factory=datetime.now)

class ApartmentFavorite(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    apartment_id: str
    created_at: datetime = Field(default_factory=datetime.now)

class TelegramChannel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel_username: str
    channel_title: Optional[str] = None
    last_parsed_message_id: Optional[int] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# Request/Response models
class CreateUserRequest(BaseModel):
    name: str
    age: int
    gender: str
    bio: Optional[str] = None
    photo_url: Optional[str] = None
    telegram_id: Optional[int] = None
    telegram_username: Optional[str] = None

class UpdatePreferencesRequest(BaseModel):
    metro_stations: List[str] = []
    radius_km: float = 5.0
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    rooms_count: Optional[List[int]] = None

class ApartmentSearchRequest(BaseModel):
    metro_stations: Optional[List[str]] = None
    radius_km: Optional[float] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    rooms_count: Optional[List[int]] = None

class UserSearchRequest(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[str] = None