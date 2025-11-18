/**
 * Faculty Profile Page
 */

'use client';

import * as React from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/lib/api/client';
import { useUserStore } from '@/lib/stores/userStore';
import { useComparisonStore } from '@/lib/stores/comparisonStore';
import type { Faculty, Publication, Grant } from '@/lib/types/api';
import { formatCurrency, formatNumber, formatDate } from '@/lib/utils/format';
import {
  Heart,
  Mail,
  ExternalLink,
  MapPin,
  Users,
  BookOpen,
  DollarSign,
  TrendingUp,
  CheckCircle,
  XCircle,
  Building,
  Calendar,
  Loader2,
  ArrowLeft,
} from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export default function FacultyProfilePage() {
  const params = useParams();
  const router = useRouter();
  const facultyId = params.id as string;

  const [faculty, setFaculty] = React.useState<Faculty | null>(null);
  const [publications, setPublications] = React.useState<Publication[]>([]);
  const [grants, setGrants] = React.useState<Grant[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [activeTab, setActiveTab] = React.useState<'overview' | 'publications' | 'grants'>('overview');

  const isSaved = useUserStore((state) => state.isFacultySaved(facultyId));
  const saveFaculty = useUserStore((state) => state.saveFaculty);
  const unsaveFaculty = useUserStore((state) => state.unsaveFaculty);
  const addToComparison = useComparisonStore((state) => state.addToComparison);
  const selectedFacultyIds = useComparisonStore((state) => state.selectedFacultyIds);

  const isInComparison = selectedFacultyIds.includes(facultyId);

  React.useEffect(() => {
    loadFacultyData();
  }, [facultyId]);

  const loadFacultyData = async () => {
    setIsLoading(true);
    try {
      const [facultyData, pubs, grantData] = await Promise.all([
        apiClient.getFaculty(facultyId),
        apiClient.getFacultyPublications(facultyId),
        apiClient.getFacultyGrants(facultyId),
      ]);

      setFaculty(facultyData);
      setPublications(pubs);
      setGrants(grantData);
    } catch (error) {
      console.error('Failed to load faculty data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveToggle = async () => {
    try {
      if (isSaved) {
        await unsaveFaculty(facultyId);
      } else {
        await saveFaculty(facultyId);
      }
    } catch (error) {
      console.error('Failed to toggle save:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!faculty) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <h2 className="text-2xl font-bold mb-4">Faculty Not Found</h2>
        <Button onClick={() => router.back()}>Go Back</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-gradient-to-b from-primary/10 to-background border-b">
        <div className="container mx-auto px-4 py-8">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.back()}
            className="mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          <div className="flex flex-col md:flex-row gap-6 items-start">
            {faculty.personalInfo.photoUrl && (
              <img
                src={faculty.personalInfo.photoUrl}
                alt={faculty.personalInfo.name}
                className="w-32 h-32 rounded-full object-cover border-4 border-background shadow-lg"
              />
            )}

            <div className="flex-1">
              <h1 className="text-3xl md:text-4xl font-bold mb-2">
                {faculty.personalInfo.name}
              </h1>
              <p className="text-xl text-muted-foreground mb-4">
                {faculty.personalInfo.title}
              </p>

              <div className="flex flex-wrap gap-4 text-sm mb-4">
                <span className="flex items-center gap-2">
                  <Building className="w-4 h-4" />
                  {faculty.affiliation.department}
                </span>
                <span className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  {faculty.affiliation.university}
                </span>
                {faculty.personalInfo.office && (
                  <span className="flex items-center gap-2">
                    <MapPin className="w-4 h-4" />
                    {faculty.personalInfo.office}
                  </span>
                )}
              </div>

              <div className="flex flex-wrap gap-2 mb-4">
                <div
                  className={cn(
                    'flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium',
                    faculty.lab.acceptingStudents
                      ? 'bg-green-100 text-green-700'
                      : 'bg-red-100 text-red-700'
                  )}
                >
                  {faculty.lab.acceptingStudents ? (
                    <CheckCircle className="w-4 h-4" />
                  ) : (
                    <XCircle className="w-4 h-4" />
                  )}
                  {faculty.lab.acceptingStudents
                    ? 'Accepting Students'
                    : 'Not Accepting'}
                </div>
                <Badge variant="secondary">
                  {faculty.metrics.careerStage} Career
                </Badge>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button onClick={handleSaveToggle} variant={isSaved ? 'default' : 'outline'}>
                  <Heart className={cn('w-4 h-4 mr-2', isSaved && 'fill-current')} />
                  {isSaved ? 'Saved' : 'Save'}
                </Button>
                <Button variant="outline">
                  <Mail className="w-4 h-4 mr-2" />
                  Contact
                </Button>
                {faculty.lab.website && (
                  <Button variant="outline" asChild>
                    <a href={faculty.lab.website} target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Lab Website
                    </a>
                  </Button>
                )}
                <Button
                  variant={isInComparison ? 'default' : 'outline'}
                  onClick={() => addToComparison(facultyId)}
                  disabled={isInComparison}
                >
                  {isInComparison ? 'In Comparison' : 'Add to Compare'}
                </Button>
              </div>
            </div>

            {/* Key Metrics */}
            <Card className="w-full md:w-64">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div>
                    <div className="text-2xl font-bold text-primary">
                      {formatCurrency(faculty.metrics.activeFunding * 1000000)}
                    </div>
                    <div className="text-xs text-muted-foreground">Active Funding</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold">{faculty.metrics.publicationCount}</div>
                    <div className="text-xs text-muted-foreground">Publications</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold">{faculty.metrics.hIndex}</div>
                    <div className="text-xs text-muted-foreground">h-index</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold">
                      {formatNumber(faculty.metrics.totalCitations)}
                    </div>
                    <div className="text-xs text-muted-foreground">Citations</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b">
        <div className="container mx-auto px-4">
          <div className="flex gap-4">
            {(['overview', 'publications', 'grants'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={cn(
                  'px-4 py-3 font-medium border-b-2 transition-colors capitalize',
                  activeTab === tab
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                )}
              >
                {tab}
                {tab === 'publications' && ` (${publications.length})`}
                {tab === 'grants' && ` (${grants.length})`}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        {activeTab === 'overview' && (
          <div className="grid md:grid-cols-3 gap-6">
            <div className="md:col-span-2 space-y-6">
              {/* Research Summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Research Focus</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{faculty.research.summary}</p>
                </CardContent>
              </Card>

              {/* Research Interests */}
              <Card>
                <CardHeader>
                  <CardTitle>Research Interests</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {faculty.research.interests.map((interest) => (
                      <Badge key={interest} variant="secondary" className="text-sm">
                        {interest}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Techniques */}
              <Card>
                <CardHeader>
                  <CardTitle>Techniques & Methods</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {faculty.research.techniques.map((tech) => (
                      <Badge key={tech} variant="outline">
                        {tech}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="space-y-6">
              {/* Lab Info */}
              <Card>
                <CardHeader>
                  <CardTitle>Lab Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Lab Size</span>
                    <span className="font-medium">{faculty.lab.size} members</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Grad Students</span>
                    <span className="font-medium">{faculty.lab.gradStudents}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Postdocs</span>
                    <span className="font-medium">{faculty.lab.postdocs}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Rotations</span>
                    <span className="font-medium capitalize">
                      {faculty.lab.rotationAvailability}
                    </span>
                  </div>
                </CardContent>
              </Card>

              {/* Model Organisms */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Organisms</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {faculty.research.organisms.map((org) => (
                      <Badge key={org} variant="outline">
                        {org}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'publications' && (
          <div className="space-y-4">
            {publications.slice(0, 20).map((pub) => (
              <Card key={pub.id}>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-2">{pub.title}</h3>
                  <p className="text-sm text-muted-foreground mb-2">
                    {pub.authors.slice(0, 5).map((a) => a.name).join(', ')}
                    {pub.authors.length > 5 && ` +${pub.authors.length - 5} more`}
                  </p>
                  <p className="text-sm text-muted-foreground mb-2">
                    {pub.journal} ({pub.year}) • {pub.citations} citations
                  </p>
                  {pub.abstract && (
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {pub.abstract}
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {activeTab === 'grants' && (
          <div className="space-y-4">
            {grants.map((grant) => (
              <Card key={grant.id}>
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={grant.isActive ? 'default' : 'secondary'}>
                          {grant.mechanism}
                        </Badge>
                        <Badge variant="outline">{grant.agency}</Badge>
                      </div>
                      <h3 className="font-semibold mb-2">{grant.title}</h3>
                      <p className="text-sm text-muted-foreground mb-2">
                        {grant.projectNumber}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-xl font-bold text-primary">
                        {formatCurrency(grant.totalAmount)}
                      </div>
                      <div className="text-xs text-muted-foreground">Total</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      {formatDate(grant.startDate)} - {formatDate(grant.endDate)}
                    </span>
                    <span
                      className={cn(
                        grant.isActive ? 'text-green-600' : 'text-muted-foreground'
                      )}
                    >
                      {grant.isActive ? 'Active' : 'Completed'}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
