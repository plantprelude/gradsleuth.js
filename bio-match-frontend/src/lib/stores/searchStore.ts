/**
 * Search State Management with Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  SearchQuery,
  SearchResult,
  SavedSearch,
} from '@/lib/types/api';
import { apiClient } from '@/lib/api/client';

interface SearchState {
  // Current search
  query: SearchQuery;
  results: SearchResult | null;
  isLoading: boolean;
  error: string | null;

  // Search history
  recentSearches: SearchQuery[];
  savedSearches: SavedSearch[];

  // Actions
  search: (query: SearchQuery) => Promise<void>;
  updateFilters: (filters: Partial<SearchQuery['filters']>) => void;
  updateQuery: (query: string) => void;
  updateSort: (sort: SearchQuery['sort']) => void;
  saveSearch: (name: string) => void;
  loadSavedSearch: (id: string) => void;
  deleteSavedSearch: (id: string) => void;
  clearResults: () => void;
  resetFilters: () => void;
}

const initialQuery: SearchQuery = {
  query: '',
  filters: {},
  sort: 'relevance',
  page: 1,
  limit: 20,
};

export const useSearchStore = create<SearchState>()(
  persist(
    (set, get) => ({
      query: initialQuery,
      results: null,
      isLoading: false,
      error: null,
      recentSearches: [],
      savedSearches: [],

      search: async (query: SearchQuery) => {
        set({ isLoading: true, error: null });

        try {
          const results = await apiClient.searchFaculty(query);

          set((state) => ({
            query,
            results,
            isLoading: false,
            recentSearches: [
              query,
              ...state.recentSearches.filter(
                (q) => JSON.stringify(q) !== JSON.stringify(query)
              ),
            ].slice(0, 10), // Keep last 10
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Search failed',
            isLoading: false,
          });
        }
      },

      updateFilters: (filters: Partial<SearchQuery['filters']>) => {
        set((state) => ({
          query: {
            ...state.query,
            filters: {
              ...state.query.filters,
              ...filters,
            },
            page: 1, // Reset to first page when filters change
          },
        }));
      },

      updateQuery: (query: string) => {
        set((state) => ({
          query: {
            ...state.query,
            query,
            page: 1,
          },
        }));
      },

      updateSort: (sort: SearchQuery['sort']) => {
        set((state) => ({
          query: {
            ...state.query,
            sort,
            page: 1,
          },
        }));
      },

      saveSearch: (name: string) => {
        const state = get();
        const newSavedSearch: SavedSearch = {
          id: `saved-${Date.now()}`,
          name,
          query: state.query,
          createdAt: new Date().toISOString(),
          alertEnabled: false,
          resultCount: state.results?.totalCount,
        };

        set((state) => ({
          savedSearches: [...state.savedSearches, newSavedSearch],
        }));
      },

      loadSavedSearch: (id: string) => {
        const state = get();
        const savedSearch = state.savedSearches.find((s) => s.id === id);
        if (savedSearch) {
          state.search(savedSearch.query);
        }
      },

      deleteSavedSearch: (id: string) => {
        set((state) => ({
          savedSearches: state.savedSearches.filter((s) => s.id !== id),
        }));
      },

      clearResults: () => {
        set({
          results: null,
          query: initialQuery,
        });
      },

      resetFilters: () => {
        set((state) => ({
          query: {
            ...state.query,
            filters: {},
            page: 1,
          },
        }));
      },
    }),
    {
      name: 'search-storage',
      partialize: (state) => ({
        recentSearches: state.recentSearches,
        savedSearches: state.savedSearches,
      }),
    }
  )
);
