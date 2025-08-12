from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
            database_name = os.environ.get('DATABASE_NAME', 'roomfinder_social')
            
            self.client = AsyncIOMotorClient(mongo_url)
            self.database = self.client[database_name]
            
            # Test connection
            await self.client.admin.command('ismaster')
            logger.info(f"Successfully connected to MongoDB: {database_name}")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Users collection indexes
            await self.users.create_index("telegram_id", unique=True, sparse=True)
            await self.users.create_index("telegram_username", unique=True, sparse=True)
            
            # Apartments collection indexes
            await self.apartments.create_index([("coordinates.latitude", 1), ("coordinates.longitude", 1)])
            await self.apartments.create_index("metro_station")
            await self.apartments.create_index("price")
            await self.apartments.create_index("is_active")
            
            # User preferences indexes
            await self.preferences.create_index("user_id", unique=True)
            
            # Likes indexes
            await self.likes.create_index([("from_user_id", 1), ("to_user_id", 1)], unique=True)
            
            # Favorites indexes
            await self.favorites.create_index([("user_id", 1), ("apartment_id", 1)], unique=True)
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    @property
    def users(self) -> AsyncIOMotorCollection:
        return self.database.users
    
    @property
    def apartments(self) -> AsyncIOMotorCollection:
        return self.database.apartments
    
    @property
    def preferences(self) -> AsyncIOMotorCollection:
        return self.database.preferences
    
    @property
    def likes(self) -> AsyncIOMotorCollection:
        return self.database.likes
    
    @property
    def favorites(self) -> AsyncIOMotorCollection:
        return self.database.favorites
    
    @property
    def channels(self) -> AsyncIOMotorCollection:
        return self.database.channels

# Global database instance
db_manager = DatabaseManager()