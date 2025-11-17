/**
 * User Dashboard
 */

'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useUserStore } from '@/lib/stores/userStore';
import { useSearchStore } from '@/lib/stores/searchStore';
import { apiClient } from '@/lib/api/client';
import type { Faculty } from '@/lib/types/api';
import { FacultyCard } from '@/components/faculty/FacultyCard';
import {
  BookmarkIcon,
  Search,
  FileText,
  TrendingUp,
  Calendar,
  Loader2,
  ArrowRight,
} from 'lucide-react';

export default function DashboardPage() {
  const router = useRouter();
  const user = useUserStore((state) => state.user);
  const isAuthenticated = useUserStore((state) => state.isAuthenticated);
  const savedFacultyIds = useUserStore((state) => state.savedFacultyIds);
  const applications = useUserStore((state) => state.applications);
  const savedSearches = useSearchStore((state) => state.savedSearches);
  const recentSearches = useSearchStore((state) => state.recentSearches);

  const [savedFaculty, setSavedFaculty] = React.useState<Faculty[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);

  React.useEffect(() => {
    if (!isAuthenticated) {
      router.push('/auth/login');
      return;
    }

    loadSavedFaculty();
  }, [isAuthenticated, savedFacultyIds]);

  const loadSavedFaculty = async () => {
    setIsLoading(true);
    try {
      const faculty = await Promise.all(
        savedFacultyIds.slice(0, 6).map((id) => apiClient.getFaculty(id))
      );
      setSavedFaculty(faculty.filter((f): f is Faculty => f !== null));
    } catch (error) {
      console.error('Failed to load saved faculty:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isAuthenticated) {
    return null;
  }

  const stats = [
    {
      label: 'Saved Faculty',
      value: savedFacultyIds.length,
      icon: BookmarkIcon,
      href: '/dashboard/saved',
    },
    {
      label: 'Saved Searches',
      value: savedSearches.length,
      icon: Search,
      href: '/dashboard/searches',
    },
    {
      label: 'Applications',
      value: applications.length,
      icon: FileText,
      href: '/dashboard/applications',
    },
    {
      label: 'Recent Searches',
      value: recentSearches.length,
      icon: TrendingUp,
      href: '/search',
    },
  ];

  const applicationsByStatus = React.useMemo(() => {
    const statusCounts = applications.reduce((acc, app) => {
      acc[app.status] = (acc[app.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    return statusCounts;
  }, [applications]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Welcome back, {user?.name}!</h1>
        <p className="text-muted-foreground">
          Track your research, manage applications, and discover new faculty matches
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card
              key={stat.label}
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => router.push(stat.href)}
            >
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">{stat.label}</p>
                    <p className="text-3xl font-bold">{stat.value}</p>
                  </div>
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Main Content Grid */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Saved Faculty */}
        <div className="md:col-span-2 space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <BookmarkIcon className="w-5 h-5" />
                Saved Faculty
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => router.push('/dashboard/saved')}
              >
                View All
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                </div>
              ) : savedFaculty.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-4">
                    No saved faculty yet
                  </p>
                  <Button onClick={() => router.push('/search')}>
                    Start Searching
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  {savedFaculty.map((faculty) => (
                    <FacultyCard
                      key={faculty.id}
                      faculty={faculty}
                      view="compact"
                      onView={() => router.push(`/faculty/${faculty.id}`)}
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Recent Searches */}
          {recentSearches.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  Recent Searches
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {recentSearches.slice(0, 5).map((search, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        router.push(`/search?q=${encodeURIComponent(search.query)}`);
                      }}
                      className="w-full text-left px-3 py-2 rounded-lg hover:bg-accent transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{search.query}</span>
                        {search.filters.acceptingStudents && (
                          <Badge variant="secondary" className="text-xs">
                            Accepting Students
                          </Badge>
                        )}
                      </div>
                      {(search.filters.universities?.length || 0) > 0 && (
                        <div className="text-xs text-muted-foreground mt-1">
                          {search.filters.universities?.join(', ')}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Application Tracker */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Application Tracker
              </CardTitle>
            </CardHeader>
            <CardContent>
              {applications.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-sm text-muted-foreground mb-3">
                    Start tracking your applications
                  </p>
                  <Button
                    size="sm"
                    onClick={() => router.push('/dashboard/applications')}
                  >
                    Add Application
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {Object.entries(applicationsByStatus).map(([status, count]) => (
                    <div
                      key={status}
                      className="flex items-center justify-between"
                    >
                      <span className="text-sm capitalize">
                        {status.replace(/-/g, ' ')}
                      </span>
                      <Badge variant="secondary">{count}</Badge>
                    </div>
                  ))}
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full mt-4"
                    onClick={() => router.push('/dashboard/applications')}
                  >
                    Manage Applications
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Saved Searches */}
          {savedSearches.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  Saved Searches
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {savedSearches.slice(0, 5).map((search) => (
                    <button
                      key={search.id}
                      onClick={() => {
                        router.push(`/search?q=${encodeURIComponent(search.query.query)}`);
                      }}
                      className="w-full text-left px-3 py-2 rounded-lg hover:bg-accent transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm">{search.name}</span>
                        {search.alertEnabled && (
                          <Badge variant="outline" className="text-xs">
                            Alert
                          </Badge>
                        )}
                      </div>
                      {search.resultCount !== undefined && (
                        <span className="text-xs text-muted-foreground">
                          {search.resultCount} results
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push('/search')}
              >
                <Search className="w-4 h-4 mr-2" />
                New Search
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push('/faculty/compare')}
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Compare Faculty
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => router.push('/dashboard/applications')}
              >
                <Calendar className="w-4 h-4 mr-2" />
                Track Application
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
