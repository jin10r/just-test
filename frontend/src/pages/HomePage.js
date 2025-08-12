import React, { useState, useEffect } from 'react';
import { 
  MapPin, 
  Heart, 
  Users, 
  Building, 
  TrendingUp,
  Sparkles,
  ArrowRight
} from 'lucide-react';
import { Link } from 'react-router-dom';
import toast from 'react-hot-toast';

const HomePage = () => {
  const [stats, setStats] = useState({
    totalApartments: 0,
    totalUsers: 0,
    totalMatches: 0
  });

  useEffect(() => {
    // Simulate loading stats
    setTimeout(() => {
      setStats({
        totalApartments: 1250,
        totalUsers: 847,
        totalMatches: 156
      });
    }, 1000);
  }, []);

  const features = [
    {
      icon: MapPin,
      title: 'Smart Map Search',
      description: 'Find apartments on an interactive Moscow map with metro station filters',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Users,
      title: 'Roommate Matching',
      description: 'Connect with like-minded people looking for rooms in your area',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Building,
      title: 'AI-Parsed Listings',
      description: 'Automatically parsed apartment listings from Telegram channels',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: Heart,
      title: 'Instant Connections',
      description: 'Like profiles and get instant matches with mutual interest',
      color: 'from-red-500 to-orange-500'
    }
  ];

  const statsCards = [
    { label: 'Active Apartments', value: stats.totalApartments, icon: Building, color: 'bg-blue-500' },
    { label: 'Active Users', value: stats.totalUsers, icon: Users, color: 'bg-purple-500' },
    { label: 'Successful Matches', value: stats.totalMatches, icon: Heart, color: 'bg-pink-500' }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center">
            {/* Main heading */}
            <div className="mb-6">
              <h1 className="text-5xl md:text-7xl font-bold text-white mb-4">
                Find Your Perfect
              </h1>
              <h1 className="text-5xl md:text-7xl font-bold gradient-text mb-6">
                Room & Roommate
              </h1>
            </div>

            <p className="text-xl md:text-2xl text-white/80 mb-8 max-w-3xl mx-auto leading-relaxed">
              The first Telegram-based social network for apartment hunting and roommate matching in Moscow. 
              Connect, discover, and find your ideal living situation.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <Link 
                to="/map"
                className="btn-primary text-lg px-8 py-4 flex items-center space-x-2"
              >
                <MapPin className="w-6 h-6" />
                <span>Explore Map</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              
              <Link 
                to="/users"
                className="btn-secondary text-lg px-8 py-4 flex items-center space-x-2"
              >
                <Users className="w-6 h-6" />
                <span>Find Roommates</span>
              </Link>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              {statsCards.map((stat, index) => {
                const Icon = stat.icon;
                return (
                  <div 
                    key={index}
                    className="glass rounded-3xl p-6 text-center animate-fade-in"
                    style={{ animationDelay: `${index * 0.2}s` }}
                  >
                    <div className={`inline-flex items-center justify-center w-16 h-16 ${stat.color} rounded-2xl mb-4`}>
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <div className="text-3xl font-bold text-white mb-2">
                      {stat.value.toLocaleString()}
                    </div>
                    <div className="text-white/80">
                      {stat.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Floating Elements */}
        <div className="absolute top-20 left-10 opacity-20">
          <Sparkles className="w-8 h-8 text-white float-animation" />
        </div>
        <div className="absolute top-40 right-10 opacity-20">
          <TrendingUp className="w-12 h-12 text-white float-animation" style={{ animationDelay: '1s' }} />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Why Choose Roomfinder?
            </h2>
            <p className="text-xl text-white/80 max-w-2xl mx-auto">
              Modern technology meets traditional apartment hunting. 
              Experience the future of room finding today.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div 
                  key={index}
                  className="glass rounded-3xl p-8 card-hover animate-slide-up"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className={`inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r ${feature.color} rounded-2xl mb-6`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  
                  <h3 className="text-2xl font-bold text-white mb-4">
                    {feature.title}
                  </h3>
                  
                  <p className="text-white/80 text-lg leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto max-w-4xl">
          <div className="glass rounded-3xl p-12 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Find Your Perfect Match?
            </h2>
            <p className="text-xl text-white/80 mb-8">
              Join thousands of users already finding their ideal living situations
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                to="/profile"
                className="btn-primary text-lg px-8 py-4"
                onClick={() => toast.success('Welcome to Roomfinder!')}
              >
                Create Profile
              </Link>
              
              <Link 
                to="/map"
                className="btn-secondary text-lg px-8 py-4"
              >
                Browse Apartments
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;