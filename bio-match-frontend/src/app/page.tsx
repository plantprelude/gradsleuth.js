/**
 * Landing Page
 */

'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { SearchBar } from '@/components/search/SearchBar';
import { useSearchStore } from '@/lib/stores/searchStore';
import {
  Search,
  Users,
  TrendingUp,
  Microscope,
  Sparkles,
  ArrowRight,
} from 'lucide-react';

export default function HomePage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = React.useState('');
  const search = useSearchStore((state) => state.search);
  const updateQuery = useSearchStore((state) => state.updateQuery);

  const handleSearch = () => {
    if (searchQuery.trim()) {
      updateQuery(searchQuery);
      router.push(`/search?q=${encodeURIComponent(searchQuery)}`);
    }
  };

  const popularSearches = [
    'CRISPR gene editing',
    'Cancer immunotherapy',
    'Neuroscience',
    'Stem cell biology',
    'Aging research',
    'Synthetic biology',
  ];

  const features = [
    {
      icon: Search,
      title: 'Intelligent Search',
      description:
        'Search by research interests, techniques, organisms, or natural language queries. Our AI understands biology.',
    },
    {
      icon: Users,
      title: 'Comprehensive Profiles',
      description:
        'View detailed faculty profiles with publications, grants, lab size, and current availability.',
    },
    {
      icon: TrendingUp,
      title: 'Match Scoring',
      description:
        'Get personalized match scores based on research alignment, funding, impact, and lab culture.',
    },
    {
      icon: Microscope,
      title: 'Real Data',
      description:
        'Data sourced from PubMed, NIH RePORTER, and institutional databases, updated regularly.',
    },
  ];

  const stats = [
    { label: 'Faculty Profiles', value: '500+' },
    { label: 'Universities', value: '50+' },
    { label: 'Research Areas', value: '100+' },
    { label: 'Publications Indexed', value: '50K+' },
  ];

  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-b from-primary/10 to-background py-20 px-4">
        <div className="container mx-auto max-w-4xl">
          <div className="text-center space-y-6">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-sm font-medium text-primary">
              <Sparkles className="w-4 h-4" />
              Find Your Perfect Research Mentor
            </div>

            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Discover Biology Faculty
              <br />
              <span className="text-primary">Matching Your Interests</span>
            </h1>

            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Search through hundreds of faculty profiles using AI-powered
              matching. Find mentors based on research interests, techniques,
              funding, and more.
            </p>

            {/* Search Bar */}
            <div className="max-w-2xl mx-auto pt-8">
              <SearchBar
                value={searchQuery}
                onChange={setSearchQuery}
                onSearch={handleSearch}
                placeholder="Try: CRISPR, cancer biology, mouse models..."
              />
            </div>

            {/* Popular Searches */}
            <div className="flex flex-wrap justify-center gap-2 pt-4">
              <span className="text-sm text-muted-foreground">Popular:</span>
              {popularSearches.map((query) => (
                <button
                  key={query}
                  onClick={() => {
                    setSearchQuery(query);
                    updateQuery(query);
                    router.push(`/search?q=${encodeURIComponent(query)}`);
                  }}
                  className="text-sm px-3 py-1 bg-secondary hover:bg-secondary/80 rounded-full transition-colors"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 border-b">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-primary">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground mt-1">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">
              Everything You Need to Find the Right Lab
            </h2>
            <p className="text-lg text-muted-foreground">
              Powerful tools to help you make informed decisions about your
              graduate research
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <Card key={feature.title} className="border-2">
                  <CardContent className="pt-6">
                    <div className="flex gap-4">
                      <div className="flex-shrink-0">
                        <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                          <Icon className="w-6 h-6 text-primary" />
                        </div>
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold mb-2">
                          {feature.title}
                        </h3>
                        <p className="text-muted-foreground">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-primary/5">
        <div className="container mx-auto max-w-4xl text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Find Your Research Home?
          </h2>
          <p className="text-lg text-muted-foreground mb-8">
            Create a free account to save searches, track applications, and get
            personalized recommendations
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" onClick={() => router.push('/auth/register')}>
              Get Started Free
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => router.push('/search')}
            >
              Browse Faculty
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
