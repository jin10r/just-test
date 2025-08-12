import asyncio
import re
import os
import logging
from typing import List, Dict, Optional, Tuple
from pyrogram import Client, types
from datetime import datetime
from models import Apartment, Coordinates, TelegramChannel
from text_analyzer import TextAnalyzer
from geopy.geocoders import Nominatim
import requests

logger = logging.getLogger(__name__)

class TelegramParser:
    def __init__(self, session_name: str = "parser_session"):
        """
        Initialize Telegram parser using existing session
        User should have already authorized the session
        """
        self.session_name = session_name
        self.client = None
        self.text_analyzer = TextAnalyzer()
        self.geocoder = Nominatim(user_agent="roomfinder_social")
        
    async def initialize(self):
        """Initialize the Telegram client"""
        try:
            # Create client without api_id and api_hash (using existing session)
            self.client = Client(self.session_name)
            await self.client.start()
            logger.info("Telegram client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            raise
    
    async def close(self):
        """Close the Telegram client"""
        if self.client:
            await self.client.stop()
    
    async def parse_channel_messages(self, channel_username: str, limit: int = 100) -> List[Apartment]:
        """
        Parse messages from a Telegram channel and extract apartment info
        """
        apartments = []
        try:
            async for message in self.client.get_chat_history(channel_username, limit=limit):
                if message.text:
                    apartment = await self._parse_message_to_apartment(message, channel_username)
                    if apartment:
                        apartments.append(apartment)
            
            logger.info(f"Parsed {len(apartments)} apartments from {channel_username}")
            return apartments
            
        except Exception as e:
            logger.error(f"Error parsing channel {channel_username}: {e}")
            return []
    
    async def _parse_message_to_apartment(self, message: types.Message, channel_username: str) -> Optional[Apartment]:
        """
        Parse a single message and extract apartment information using BERT
        """
        try:
            text = message.text
            if not text:
                return None
            
            # Use BERT to extract structured information
            parsed_info = await self.text_analyzer.extract_apartment_info(text)
            
            if not parsed_info or not parsed_info.get('is_rent_announcement'):
                return None
            
            # Extract coordinates from text
            coordinates = self._extract_coordinates(text)
            
            # If no coordinates found, try geocoding the address
            if not coordinates and parsed_info.get('address'):
                coordinates = await self._geocode_address(parsed_info['address'])
            
            # Extract photos
            photos = []
            if message.photo:
                photos.append(f"telegram_photo_{message.id}")
            
            apartment = Apartment(
                title=parsed_info.get('title', text[:100] + '...' if len(text) > 100 else text),
                description=text,
                price=parsed_info.get('price'),
                address=parsed_info.get('address', ''),
                coordinates=coordinates,
                metro_station=parsed_info.get('metro_station'),
                rooms_count=parsed_info.get('rooms_count'),
                area=parsed_info.get('area'),
                floor=parsed_info.get('floor'),
                total_floors=parsed_info.get('total_floors'),
                photos=photos,
                contact_info=parsed_info.get('contact_info'),
                telegram_channel=channel_username,
                message_id=message.id
            )
            
            return apartment
            
        except Exception as e:
            logger.error(f"Error parsing message {message.id}: {e}")
            return None
    
    def _extract_coordinates(self, text: str) -> Optional[Coordinates]:
        """
        Extract coordinates from text using regex patterns
        """
        # Pattern for coordinates in various formats
        coord_patterns = [
            r'(\d+\.\d+),\s*(\d+\.\d+)',  # 55.7558, 37.6173
            r'(\d+\.\d+)\s+(\d+\.\d+)',   # 55.7558 37.6173
            r'lat:?\s*(\d+\.\d+).*?lon:?\s*(\d+\.\d+)',  # lat: 55.7558 lon: 37.6173
        ]
        
        for pattern in coord_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    # Validate coordinates are in Moscow area
                    if 55.0 <= lat <= 56.5 and 36.0 <= lon <= 38.5:
                        return Coordinates(latitude=lat, longitude=lon)
                except ValueError:
                    continue
        
        return None
    
    async def _geocode_address(self, address: str) -> Optional[Coordinates]:
        """
        Geocode address to get coordinates using Nominatim
        """
        try:
            # Add Moscow context to improve geocoding accuracy
            full_address = f"{address}, Moscow, Russia"
            location = self.geocoder.geocode(full_address, timeout=10)
            
            if location:
                return Coordinates(latitude=location.latitude, longitude=location.longitude)
                
        except Exception as e:
            logger.error(f"Geocoding failed for address '{address}': {e}")
        
        return None
    
    async def get_channel_info(self, channel_username: str) -> Optional[Dict]:
        """
        Get information about a Telegram channel
        """
        try:
            chat = await self.client.get_chat(channel_username)
            return {
                'id': chat.id,
                'title': chat.title,
                'username': chat.username,
                'members_count': chat.members_count if hasattr(chat, 'members_count') else None,
                'description': chat.description
            }
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_username}: {e}")
            return None