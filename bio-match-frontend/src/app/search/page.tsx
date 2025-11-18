/**
 * Search Page
 */

'use client';

import * as React from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { SearchBar } from '@/components/search/SearchBar';
import { SearchFilters } from '@/components/search/SearchFilters';
import { FacultyCard } from '@/components/faculty/FacultyCard';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useSearchStore } from '@/lib/stores/searchStore';
import { Loader2, SlidersHorizontal, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export default function SearchPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [showFilters, setShowFilters] = React.useState(true);

  const {
    query,
    results,
    isLoading,
    error,
    search,
    updateQuery,
    updateFilters,
    updateSort,
    resetFilters,
  } = useSearchStore();

  // Initialize from URL params
  React.useEffect(() => {
    const q = searchParams.get('q');
    if (q && q !== query.query) {
      updateQuery(q);
      search({ ...query, query: q });
    }
  }, [searchParams]);

  const handleSearch = () => {
    search(query);
    router.push(`/search?q=${encodeURIComponent(query.query)}`);
  };

  const handleViewFaculty = (facultyId: string) => {
    router.push(`/faculty/${facultyId}`);
  };

  const sortOptions = [
    { value: 'relevance', label: 'Most Relevant' },
    { value: 'funding-desc', label: 'Highest Funding' },
    { value: 'publications-desc', label: 'Most Publications' },
    { value: 'h-index-desc', label: 'Highest H-Index' },
    { value: 'name-asc', label: 'Name (A-Z)' },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Search Header */}
      <div className="mb-6">
        <SearchBar
          value={query.query}
          onChange={updateQuery}
          onSearch={handleSearch}
        />

        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className="md:hidden"
            >
              <SlidersHorizontal className="w-4 h-4 mr-2" />
              Filters
            </Button>

            {results && (
              <span className="text-sm text-muted-foreground">
                {results.totalCount} results
                {results.queryTime && (
                  <> in {results.queryTime}ms</>
                )}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground hidden sm:block">
              Sort by:
            </span>
            <select
              value={query.sort || 'relevance'}
              onChange={(e) =>
                updateSort(
                  e.target.value as typeof query.sort
                )
              }
              className="text-sm border rounded-md px-3 py-1.5 bg-background"
            >
              {sortOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Active Filters Display */}
        {(query.filters.universities?.length ||
          query.filters.departments?.length ||
          query.filters.acceptingStudents !== undefined) && (
          <div className="flex flex-wrap items-center gap-2 mt-4">
            <span className="text-sm text-muted-foreground">Filters:</span>
            {query.filters.universities?.map((uni) => (
              <Badge key={uni} variant="secondary">
                {uni}
                <button
                  onClick={() =>
                    updateFilters({
                      universities: query.filters.universities?.filter(
                        (u) => u !== uni
                      ),
                    })
                  }
                  className="ml-1"
                >
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            ))}
            {query.filters.departments?.map((dept) => (
              <Badge key={dept} variant="secondary">
                {dept}
                <button
                  onClick={() =>
                    updateFilters({
                      departments: query.filters.departments?.filter(
                        (d) => d !== dept
                      ),
                    })
                  }
                  className="ml-1"
                >
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            ))}
            {query.filters.acceptingStudents && (
              <Badge variant="secondary">
                Accepting Students
                <button
                  onClick={() =>
                    updateFilters({ acceptingStudents: undefined })
                  }
                  className="ml-1"
                >
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={resetFilters}
              className="h-6 text-xs"
            >
              Clear all
            </Button>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Filters Sidebar */}
        <div
          className={cn(
            'md:col-span-1',
            !showFilters && 'hidden md:block'
          )}
        >
          <div className="sticky top-20">
            <SearchFilters
              filters={query.filters}
              facets={results?.facets}
              onChange={updateFilters}
              onReset={resetFilters}
            />
          </div>
        </div>

        {/* Results */}
        <div className="md:col-span-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
          ) : error ? (
            <div className="text-center py-20">
              <p className="text-red-500 mb-4">{error}</p>
              <Button onClick={handleSearch}>Try Again</Button>
            </div>
          ) : results?.faculty.length === 0 ? (
            <div className="text-center py-20">
              <h3 className="text-xl font-semibold mb-2">No results found</h3>
              <p className="text-muted-foreground mb-4">
                Try adjusting your search or filters
              </p>
              <Button onClick={resetFilters} variant="outline">
                Clear Filters
              </Button>
            </div>
          ) : (
            <div className="space-y-6">
              {results?.faculty.map((faculty) => (
                <FacultyCard
                  key={faculty.id}
                  faculty={faculty}
                  view="detailed"
                  onView={() => handleViewFaculty(faculty.id)}
                />
              ))}

              {/* Pagination */}
              {results && results.totalCount > query.limit! && (
                <div className="flex justify-center gap-2 mt-8">
                  <Button
                    variant="outline"
                    disabled={query.page === 1}
                    onClick={() =>
                      search({ ...query, page: query.page! - 1 })
                    }
                  >
                    Previous
                  </Button>
                  <span className="flex items-center px-4 text-sm text-muted-foreground">
                    Page {query.page} of{' '}
                    {Math.ceil(results.totalCount / query.limit!)}
                  </span>
                  <Button
                    variant="outline"
                    disabled={
                      query.page! >= Math.ceil(results.totalCount / query.limit!)
                    }
                    onClick={() =>
                      search({ ...query, page: query.page! + 1 })
                    }
                  >
                    Next
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
