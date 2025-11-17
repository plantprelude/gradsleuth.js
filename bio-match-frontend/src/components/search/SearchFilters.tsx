/**
 * SearchFilters Component
 * Advanced filtering for faculty search
 */

'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { SearchFilters as Filters, FacetValue } from '@/lib/types/api';
import { X, Filter, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface SearchFiltersProps {
  filters: Filters;
  facets?: {
    universities: FacetValue[];
    departments: FacetValue[];
    techniques: FacetValue[];
    organisms: FacetValue[];
    states: FacetValue[];
  };
  onChange: (filters: Partial<Filters>) => void;
  onReset: () => void;
  className?: string;
}

export function SearchFilters({
  filters,
  facets,
  onChange,
  onReset,
  className,
}: SearchFiltersProps) {
  const [expandedSections, setExpandedSections] = React.useState<Set<string>>(
    new Set(['universities', 'accepting'])
  );

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  const activeFilterCount = React.useMemo(() => {
    let count = 0;
    if (filters.universities?.length) count += filters.universities.length;
    if (filters.departments?.length) count += filters.departments.length;
    if (filters.states?.length) count += filters.states.length;
    if (filters.techniques?.length) count += filters.techniques.length;
    if (filters.organisms?.length) count += filters.organisms.length;
    if (filters.acceptingStudents !== undefined) count += 1;
    if (filters.fundingMin) count += 1;
    return count;
  }, [filters]);

  const FilterSection = ({
    title,
    id,
    children,
  }: {
    title: string;
    id: string;
    children: React.ReactNode;
  }) => {
    const isExpanded = expandedSections.has(id);
    return (
      <div className="border-b last:border-b-0">
        <button
          onClick={() => toggleSection(id)}
          className="w-full flex items-center justify-between py-3 px-4 hover:bg-accent/50 transition-colors"
        >
          <span className="font-semibold text-sm">{title}</span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>
        {isExpanded && <div className="px-4 pb-3">{children}</div>}
      </div>
    );
  };

  const MultiSelectFilter = ({
    options,
    selected = [],
    onChange: onSelectionChange,
  }: {
    options: FacetValue[];
    selected?: string[];
    onChange: (values: string[]) => void;
  }) => {
    const handleToggle = (value: string) => {
      if (selected.includes(value)) {
        onSelectionChange(selected.filter((v) => v !== value));
      } else {
        onSelectionChange([...selected, value]);
      }
    };

    return (
      <div className="space-y-2">
        {options.slice(0, 8).map((option) => (
          <label
            key={option.value}
            className="flex items-center gap-2 cursor-pointer hover:bg-accent/30 p-1 rounded"
          >
            <input
              type="checkbox"
              checked={selected.includes(option.value)}
              onChange={() => handleToggle(option.value)}
              className="w-4 h-4 rounded border-gray-300"
            />
            <span className="text-sm flex-1">{option.value}</span>
            <span className="text-xs text-muted-foreground">
              {option.count}
            </span>
          </label>
        ))}
      </div>
    );
  };

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <Filter className="w-4 h-4" />
          Filters
          {activeFilterCount > 0 && (
            <Badge variant="secondary" className="ml-2">
              {activeFilterCount}
            </Badge>
          )}
        </CardTitle>
        {activeFilterCount > 0 && (
          <Button variant="ghost" size="sm" onClick={onReset}>
            <X className="w-4 h-4 mr-1" />
            Clear
          </Button>
        )}
      </CardHeader>
      <CardContent className="p-0">
        <FilterSection title="Accepting Students" id="accepting">
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer hover:bg-accent/30 p-1 rounded">
              <input
                type="radio"
                checked={filters.acceptingStudents === true}
                onChange={() => onChange({ acceptingStudents: true })}
                className="w-4 h-4"
              />
              <span className="text-sm">Accepting students only</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer hover:bg-accent/30 p-1 rounded">
              <input
                type="radio"
                checked={filters.acceptingStudents === undefined}
                onChange={() => onChange({ acceptingStudents: undefined })}
                className="w-4 h-4"
              />
              <span className="text-sm">All faculty</span>
            </label>
          </div>
        </FilterSection>

        {facets?.universities && (
          <FilterSection title="University" id="universities">
            <MultiSelectFilter
              options={facets.universities}
              selected={filters.universities}
              onChange={(universities) => onChange({ universities })}
            />
          </FilterSection>
        )}

        {facets?.departments && (
          <FilterSection title="Department" id="departments">
            <MultiSelectFilter
              options={facets.departments}
              selected={filters.departments}
              onChange={(departments) => onChange({ departments })}
            />
          </FilterSection>
        )}

        {facets?.states && (
          <FilterSection title="State" id="states">
            <MultiSelectFilter
              options={facets.states}
              selected={filters.states}
              onChange={(states) => onChange({ states })}
            />
          </FilterSection>
        )}

        {facets?.techniques && (
          <FilterSection title="Techniques" id="techniques">
            <MultiSelectFilter
              options={facets.techniques}
              selected={filters.techniques}
              onChange={(techniques) => onChange({ techniques })}
            />
          </FilterSection>
        )}

        {facets?.organisms && (
          <FilterSection title="Model Organisms" id="organisms">
            <MultiSelectFilter
              options={facets.organisms}
              selected={filters.organisms}
              onChange={(organisms) => onChange({ organisms })}
            />
          </FilterSection>
        )}

        <FilterSection title="Funding" id="funding">
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground">
              Minimum Active Funding (in millions)
            </label>
            <input
              type="range"
              min="0"
              max="10"
              step="0.5"
              value={filters.fundingMin || 0}
              onChange={(e) =>
                onChange({ fundingMin: parseFloat(e.target.value) })
              }
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>$0M</span>
              <span className="font-medium text-foreground">
                ${filters.fundingMin || 0}M+
              </span>
              <span>$10M+</span>
            </div>
          </div>
        </FilterSection>

        <FilterSection title="Career Stage" id="career">
          <div className="space-y-2">
            {['early', 'mid', 'senior'].map((stage) => (
              <label
                key={stage}
                className="flex items-center gap-2 cursor-pointer hover:bg-accent/30 p-1 rounded"
              >
                <input
                  type="checkbox"
                  checked={filters.careerStage?.includes(
                    stage as 'early' | 'mid' | 'senior'
                  )}
                  onChange={(e) => {
                    const current = filters.careerStage || [];
                    const updated = e.target.checked
                      ? [...current, stage as 'early' | 'mid' | 'senior']
                      : current.filter((s) => s !== stage);
                    onChange({ careerStage: updated });
                  }}
                  className="w-4 h-4 rounded border-gray-300"
                />
                <span className="text-sm capitalize">{stage} Career</span>
              </label>
            ))}
          </div>
        </FilterSection>
      </CardContent>
    </Card>
  );
}
