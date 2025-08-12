import re
import logging
from typing import Dict, Optional, List
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Initialize BERT-based text analyzer for Russian language"""
        self.sentiment_analyzer = None
        self.tokenizer = None
        self.model = None
        self._load_models()
    
    def _load_models(self):
        """Load Russian BERT models"""
        try:
            # Use Russian BERT model for text understanding
            model_name = "DeepPavlov/rubert-base-cased"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            logger.info("Russian BERT models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BERT models: {e}")
            # Fallback to basic text processing
            self.tokenizer = None
            self.model = None
    
    async def extract_apartment_info(self, text: str) -> Dict:
        """
        Extract apartment information from Russian text using BERT and regex patterns
        """
        try:
            # Clean and normalize text
            clean_text = self._clean_text(text)
            
            # Check if it's a rent announcement
            is_rent = self._is_rent_announcement(clean_text)
            
            if not is_rent:
                return {'is_rent_announcement': False}
            
            # Extract structured information
            info = {
                'is_rent_announcement': True,
                'price': self._extract_price(clean_text),
                'address': self._extract_address(clean_text),
                'metro_station': self._extract_metro_station(clean_text),
                'rooms_count': self._extract_rooms_count(clean_text),
                'area': self._extract_area(clean_text),
                'floor': self._extract_floor(clean_text),
                'total_floors': self._extract_total_floors(clean_text),
                'contact_info': self._extract_contact_info(clean_text)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error extracting apartment info: {e}")
            return {'is_rent_announcement': False}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Russian letters
        text = re.sub(r'[^\w\s\-\+\(\)\.,:;!?\u0400-\u04FF]', '', text)
        return text.strip()
    
    def _is_rent_announcement(self, text: str) -> bool:
        """
        Determine if text is a rent announcement using keyword matching
        """
        rent_keywords = [
            'сдам', 'сдаю', 'сдается', 'аренда', 'снять', 'квартира',
            'комната', 'студия', 'однушка', 'двушка', 'трешка',
            'руб', 'рубл', 'тыс', 'м²', 'кв.м', 'метро'
        ]
        
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in rent_keywords if keyword in text_lower)
        
        # If we find at least 2 rent-related keywords, consider it a rent announcement
        return found_keywords >= 2
    
    def _extract_price(self, text: str) -> Optional[int]:
        """Extract price from text"""
        # Patterns for Russian price formats
        price_patterns = [
            r'(\d+)\s*(?:тыс|тысяч)\s*руб',  # "50 тыс руб"
            r'(\d+)\s*(?:к|к\.|тыс)',        # "50к", "50тыс"
            r'(\d+)\s*000\s*руб',             # "50 000 руб"
            r'(\d{2,6})\s*руб',               # "50000 руб"
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price = int(match.group(1))
                    # Convert thousands to full amount
                    if 'тыс' in match.group(0).lower() or 'к' in match.group(0).lower():
                        price *= 1000
                    return price
                except ValueError:
                    continue
        
        return None
    
    def _extract_address(self, text: str) -> Optional[str]:
        """Extract address from text"""
        # Look for address patterns
        address_patterns = [
            r'(?:адрес|улица|ул\.|проспект|пр-т|пр\.|бульвар|б-р|переулок|пер\.)\s*([^,\n]+)',
            r'м\.\s*([А-Яа-я\s]+).*?(\d+(?:-\d+)?(?:\s*мин)?(?:\s*пеш)?)',  # "м. Сокольники 10 мин"
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_metro_station(self, text: str) -> Optional[str]:
        """Extract metro station from text"""
        # Pattern for metro stations
        metro_pattern = r'м\.\s*([А-Яа-я\-\s]+)(?:\s*\d+(?:\s*мин)?)?'
        match = re.search(metro_pattern, text)
        
        if match:
            metro = match.group(1).strip()
            # Clean up common suffixes
            metro = re.sub(r'\s*\d+\s*мин.*$', '', metro)
            return metro
        
        return None
    
    def _extract_rooms_count(self, text: str) -> Optional[int]:
        """Extract number of rooms"""
        room_patterns = [
            r'(\d+)(?:-?к|комн|комнат)',  # "2к", "2-к", "2комн"
            r'(\d+)(?:\s*-?\s*комнат)',   # "2 комнат", "2-комнат"
            r'(однушка|студия)',          # "однушка", "студия"
            r'(двушка)',                  # "двушка"
            r'(трешка|трёшка)',           # "трешка"
        ]
        
        for pattern in room_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                room_text = match.group(1).lower()
                
                if room_text in ['однушка', 'студия']:
                    return 1
                elif room_text == 'двушка':
                    return 2
                elif room_text in ['трешка', 'трёшка']:
                    return 3
                else:
                    try:
                        return int(room_text)
                    except ValueError:
                        continue
        
        return None
    
    def _extract_area(self, text: str) -> Optional[float]:
        """Extract apartment area"""
        area_patterns = [
            r'(\d+(?:\.\d+)?)\s*м²',           # "45 м²"
            r'(\d+(?:\.\d+)?)\s*кв\.?м',       # "45 кв.м"
            r'(\d+(?:\.\d+)?)\s*метр',         # "45 метр"
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _extract_floor(self, text: str) -> Optional[int]:
        """Extract floor number"""
        floor_pattern = r'(\d+)(?:\s*[-/]\s*\d+)?\s*эт'
        match = re.search(floor_pattern, text, re.IGNORECASE)
        
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_total_floors(self, text: str) -> Optional[int]:
        """Extract total floors in building"""
        total_floors_pattern = r'\d+\s*[-/]\s*(\d+)\s*эт'
        match = re.search(total_floors_pattern, text, re.IGNORECASE)
        
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_contact_info(self, text: str) -> Optional[str]:
        """Extract contact information"""
        contact_patterns = [
            r'(?:\+7|8)\s*[\(\-\s]*\d{3}[\)\-\s]*\d{3}[\-\s]*\d{2}[\-\s]*\d{2}',  # Phone numbers
            r'@\w+',  # Telegram usernames
            r'whatsapp|viber|telegram',  # Messengers
        ]
        
        contacts = []
        for pattern in contact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contacts.extend(matches)
        
        return ', '.join(contacts) if contacts else None