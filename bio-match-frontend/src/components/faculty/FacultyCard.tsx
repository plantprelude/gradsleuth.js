/**
 * Faculty Card Component
 * Displays faculty information in various views
 */

'use client';

import * as React from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import type { Faculty, MatchScore } from '@/lib/types/api';
import { formatCurrency, formatNumber } from '@/lib/utils/format';
import { cn } from '@/lib/utils/cn';
import {
  Heart,
  Mail,
  ExternalLink,
  TrendingUp,
  Users,
  BookOpen,
  DollarSign,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react';
import { useUserStore } from '@/lib/stores/userStore';
import { useComparisonStore } from '@/lib/stores/comparisonStore';

interface FacultyCardProps {
  faculty: Faculty;
  matchScore?: MatchScore;
  view?: 'compact' | 'detailed' | 'comparison';
  onView?: () => void;
  className?: string;
}

export function FacultyCard({
  faculty,
  matchScore,
  view = 'detailed',
  onView,
  className,
}: FacultyCardProps) {
  const isSaved = useUserStore((state) =>
    state.isFacultySaved(faculty.id)
  );
  const saveFaculty = useUserStore((state) => state.saveFaculty);
  const unsaveFaculty = useUserStore((state) => state.unsaveFaculty);
  const addToComparison = useComparisonStore((state) => state.addToComparison);
  const removeFromComparison = useComparisonStore(
    (state) => state.removeFromComparison
  );
  const selectedFacultyIds = useComparisonStore(
    (state) => state.selectedFacultyIds
  );

  const isInComparison = selectedFacultyIds.includes(faculty.id);

  const handleSaveToggle = async () => {
    try {
      if (isSaved) {
        await unsaveFaculty(faculty.id);
      } else {
        await saveFaculty(faculty.id);
      }
    } catch (error) {
      console.error('Failed to toggle save:', error);
    }
  };

  const handleComparisonToggle = () => {
    if (isInComparison) {
      removeFromComparison(faculty.id);
    } else {
      addToComparison(faculty.id);
    }
  };

  if (view === 'compact') {
    return (
      <Card
        className={cn(
          'hover:shadow-md transition-shadow cursor-pointer',
          className
        )}
        onClick={onView}
      >
        <CardHeader className="p-4">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="font-semibold text-lg">
                {faculty.personalInfo.name}
              </h3>
              <p className="text-sm text-muted-foreground">
                {faculty.personalInfo.title}
              </p>
              <p className="text-sm text-muted-foreground">
                {faculty.affiliation.university}
              </p>
            </div>
            {matchScore && (
              <div className="text-right">
                <div className="text-2xl font-bold text-primary">
                  {matchScore.overallScore}
                </div>
                <div className="text-xs text-muted-foreground">Match</div>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="flex flex-wrap gap-1 mb-2">
            {faculty.research.interests.slice(0, 3).map((interest) => (
              <Badge key={interest} variant="secondary" className="text-xs">
                {interest}
              </Badge>
            ))}
            {faculty.research.interests.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{faculty.research.interests.length - 3}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <DollarSign className="w-4 h-4" />
              {formatCurrency(faculty.metrics.activeFunding * 1000000)}
            </span>
            <span className="flex items-center gap-1">
              <BookOpen className="w-4 h-4" />
              {faculty.metrics.publicationCount}
            </span>
            <span className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              h={faculty.metrics.hIndex}
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn('hover:shadow-lg transition-shadow', className)}>
      <CardHeader>
        <div className="flex items-start gap-4">
          {faculty.personalInfo.photoUrl && (
            <img
              src={faculty.personalInfo.photoUrl}
              alt={faculty.personalInfo.name}
              className="w-20 h-20 rounded-full object-cover"
            />
          )}
          <div className="flex-1">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-xl font-bold">
                  {faculty.personalInfo.name}
                </h3>
                <p className="text-muted-foreground">
                  {faculty.personalInfo.title}
                </p>
                <p className="text-sm text-muted-foreground">
                  {faculty.affiliation.department} •{' '}
                  {faculty.affiliation.university}
                </p>
              </div>
              {matchScore && (
                <div className="text-center bg-primary/10 rounded-lg p-3">
                  <div className="text-3xl font-bold text-primary">
                    {matchScore.overallScore}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Match Score
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Lab Status */}
        <div className="flex items-center gap-4 text-sm">
          <div
            className={cn(
              'flex items-center gap-1',
              faculty.lab.acceptingStudents
                ? 'text-green-600'
                : 'text-red-600'
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
          <div className="flex items-center gap-1 text-muted-foreground">
            <Users className="w-4 h-4" />
            {faculty.lab.size} lab members
          </div>
          {faculty.lab.rotationAvailability && (
            <div className="flex items-center gap-1 text-muted-foreground">
              <Clock className="w-4 h-4" />
              Rotations: {faculty.lab.rotationAvailability}
            </div>
          )}
        </div>

        {/* Research Summary */}
        <div>
          <h4 className="font-semibold mb-2">Research Focus</h4>
          <p className="text-sm text-muted-foreground line-clamp-2">
            {faculty.research.summary}
          </p>
        </div>

        {/* Research Areas */}
        <div>
          <h4 className="font-semibold mb-2">Research Interests</h4>
          <div className="flex flex-wrap gap-2">
            {faculty.research.interests.map((interest) => (
              <Badge key={interest} variant="secondary">
                {interest}
              </Badge>
            ))}
          </div>
        </div>

        {/* Techniques & Organisms */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold text-sm mb-1">Techniques</h4>
            <div className="flex flex-wrap gap-1">
              {faculty.research.techniques.slice(0, 4).map((tech) => (
                <Badge key={tech} variant="outline" className="text-xs">
                  {tech}
                </Badge>
              ))}
              {faculty.research.techniques.length > 4 && (
                <Badge variant="outline" className="text-xs">
                  +{faculty.research.techniques.length - 4}
                </Badge>
              )}
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-sm mb-1">Organisms</h4>
            <div className="flex flex-wrap gap-1">
              {faculty.research.organisms.map((org) => (
                <Badge key={org} variant="outline" className="text-xs">
                  {org}
                </Badge>
              ))}
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-4 py-3 border-t">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">
              {formatCurrency(faculty.metrics.activeFunding * 1000000)}
            </div>
            <div className="text-xs text-muted-foreground">Funding</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">
              {faculty.metrics.publicationCount}
            </div>
            <div className="text-xs text-muted-foreground">Publications</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{faculty.metrics.hIndex}</div>
            <div className="text-xs text-muted-foreground">h-index</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">
              {formatNumber(faculty.metrics.totalCitations)}
            </div>
            <div className="text-xs text-muted-foreground">Citations</div>
          </div>
        </div>

        {/* Match Score Breakdown */}
        {matchScore && (
          <div className="space-y-2 pt-3 border-t">
            <h4 className="font-semibold text-sm">Match Breakdown</h4>
            <div className="space-y-1">
              {Object.entries(matchScore.breakdown).map(([key, value]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground w-32 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </span>
                  <div className="flex-1 bg-secondary rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all"
                      style={{ width: `${value}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium w-8">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-3">
          <Button onClick={onView} className="flex-1">
            <ExternalLink className="w-4 h-4 mr-2" />
            View Profile
          </Button>
          <Button
            variant={isSaved ? 'default' : 'outline'}
            size="icon"
            onClick={handleSaveToggle}
          >
            <Heart
              className={cn('w-4 h-4', isSaved && 'fill-current')}
            />
          </Button>
          <Button variant="outline" size="icon">
            <Mail className="w-4 h-4" />
          </Button>
          <Button
            variant={isInComparison ? 'default' : 'outline'}
            onClick={handleComparisonToggle}
          >
            {isInComparison ? 'Remove' : 'Compare'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
