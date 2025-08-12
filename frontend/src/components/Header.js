import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Map, 
  Users, 
  User, 
  Heart,
  Search,
  Settings
} from 'lucide-react';

const Header = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/map', icon: Map, label: 'Map' },
    { path: '/users', icon: Users, label: 'Users' },
    { path: '/profile', icon: User, label: 'Profile' },
  ];

  return (
    <header className="header-glass sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="bg-white/20 backdrop-blur-sm rounded-2xl p-2">
              <Home className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white hidden sm:block">
              Roomfinder
            </span>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`nav-link flex items-center space-x-2 px-4 py-2 rounded-2xl transition-all duration-300 ${
                    isActive 
                      ? 'bg-white/20 backdrop-blur-sm text-white' 
                      : 'hover:bg-white/10'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Search & Settings */}
          <div className="flex items-center space-x-2">
            <button className="p-2 rounded-2xl bg-white/10 hover:bg-white/20 transition-all duration-300">
              <Search className="w-5 h-5 text-white" />
            </button>
            <button className="p-2 rounded-2xl bg-white/10 hover:bg-white/20 transition-all duration-300">
              <Settings className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;