/**
 * Faculty Comparison State Management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Faculty, ComparisonData } from '@/lib/types/api';
import { apiClient } from '@/lib/api/client';

interface ComparisonState {
  selectedFacultyIds: string[];
  facultyData: Map<string, Faculty>;
  comparisonData: ComparisonData | null;
  isLoading: boolean;

  // Actions
  addToComparison: (facultyId: string) => void;
  removeFromComparison: (facultyId: string) => void;
  clearComparison: () => void;
  loadFacultyData: () => Promise<void>;
  canAddMore: () => boolean;
}

const MAX_COMPARISON = 4;

export const useComparisonStore = create<ComparisonState>()(
  persist(
    (set, get) => ({
      selectedFacultyIds: [],
      facultyData: new Map(),
      comparisonData: null,
      isLoading: false,

      addToComparison: (facultyId: string) => {
        const state = get();
        if (state.selectedFacultyIds.length >= MAX_COMPARISON) {
          console.warn(`Maximum ${MAX_COMPARISON} faculty can be compared`);
          return;
        }

        if (state.selectedFacultyIds.includes(facultyId)) {
          return;
        }

        set({
          selectedFacultyIds: [...state.selectedFacultyIds, facultyId],
        });

        // Automatically load faculty data
        state.loadFacultyData();
      },

      removeFromComparison: (facultyId: string) => {
        set((state) => {
          const newFacultyData = new Map(state.facultyData);
          newFacultyData.delete(facultyId);

          return {
            selectedFacultyIds: state.selectedFacultyIds.filter(
              (id) => id !== facultyId
            ),
            facultyData: newFacultyData,
            comparisonData: null,
          };
        });
      },

      clearComparison: () => {
        set({
          selectedFacultyIds: [],
          facultyData: new Map(),
          comparisonData: null,
        });
      },

      loadFacultyData: async () => {
        const state = get();
        set({ isLoading: true });

        try {
          const newFacultyData = new Map(state.facultyData);

          // Load any faculty that aren't already loaded
          for (const facultyId of state.selectedFacultyIds) {
            if (!newFacultyData.has(facultyId)) {
              const faculty = await apiClient.getFaculty(facultyId);
              if (faculty) {
                newFacultyData.set(facultyId, faculty);
              }
            }
          }

          // Generate comparison data
          const facultyArray = Array.from(newFacultyData.values());
          const comparisonData: ComparisonData = {
            faculty: facultyArray,
            metrics: generateComparisonMetrics(facultyArray),
            charts: generateComparisonCharts(facultyArray),
          };

          set({
            facultyData: newFacultyData,
            comparisonData,
            isLoading: false,
          });
        } catch (error) {
          console.error('Failed to load faculty data:', error);
          set({ isLoading: false });
        }
      },

      canAddMore: () => {
        return get().selectedFacultyIds.length < MAX_COMPARISON;
      },
    }),
    {
      name: 'comparison-storage',
      partialize: (state) => ({
        selectedFacultyIds: state.selectedFacultyIds,
      }),
    }
  )
);

// Helper functions for generating comparison data
function generateComparisonMetrics(faculty: Faculty[]) {
  const metrics: ComparisonData['metrics'] = {};

  faculty.forEach((f) => {
    metrics[f.id] = {
      strengths: [],
      weaknesses: [],
      unique: [],
    };

    // Find strengths (above average)
    const avgFunding =
      faculty.reduce((sum, faculty) => sum + faculty.metrics.activeFunding, 0) /
      faculty.length;
    const avgPubs =
      faculty.reduce((sum, faculty) => sum + faculty.metrics.publicationCount, 0) /
      faculty.length;
    const avgHIndex =
      faculty.reduce((sum, faculty) => sum + faculty.metrics.hIndex, 0) /
      faculty.length;

    if (f.metrics.activeFunding > avgFunding * 1.2) {
      metrics[f.id].strengths.push(
        `High funding ($${f.metrics.activeFunding.toFixed(1)}M)`
      );
    }
    if (f.metrics.publicationCount > avgPubs * 1.2) {
      metrics[f.id].strengths.push(`Highly productive (${f.metrics.publicationCount} pubs)`);
    }
    if (f.metrics.hIndex > avgHIndex * 1.2) {
      metrics[f.id].strengths.push(`High impact (h-index: ${f.metrics.hIndex})`);
    }
    if (f.lab.acceptingStudents) {
      metrics[f.id].strengths.push('Currently accepting students');
    }

    // Find unique research areas
    const allAreas = faculty.flatMap((faculty) => faculty.research.interests);
    const uniqueAreas = f.research.interests.filter(
      (area) => allAreas.filter((a) => a === area).length === 1
    );
    metrics[f.id].unique = uniqueAreas;

    // Weaknesses
    if (f.metrics.activeFunding < avgFunding * 0.8) {
      metrics[f.id].weaknesses.push('Lower than average funding');
    }
    if (!f.lab.acceptingStudents) {
      metrics[f.id].weaknesses.push('Not currently accepting students');
    }
  });

  return metrics;
}

function generateComparisonCharts(faculty: Faculty[]) {
  return [
    {
      type: 'radar' as const,
      title: 'Overall Comparison',
      data: {
        labels: [
          'Funding',
          'Publications',
          'H-Index',
          'Lab Size',
          'Accepting Students',
        ],
        datasets: faculty.map((f) => ({
          label: f.personalInfo.name,
          data: [
            f.metrics.activeFunding * 10, // Normalize to 0-100
            Math.min(100, f.metrics.publicationCount),
            f.metrics.hIndex * 2,
            f.lab.size * 5,
            f.lab.acceptingStudents ? 100 : 0,
          ],
        })),
      },
    },
    {
      type: 'bar' as const,
      title: 'Publication Metrics',
      data: {
        labels: faculty.map((f) => f.personalInfo.lastName),
        datasets: [
          {
            label: 'Total Publications',
            data: faculty.map((f) => f.metrics.publicationCount),
          },
          {
            label: 'Recent (2 years)',
            data: faculty.map((f) => f.metrics.recentPublications),
          },
        ],
      },
    },
  ];
}
