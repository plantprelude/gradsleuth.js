/**
 * User State Management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, UserPreferences, Application } from '@/lib/types/api';
import { apiClient } from '@/lib/api/client';

interface UserState {
  user: User | null;
  isAuthenticated: boolean;
  savedFacultyIds: string[];
  applications: Application[];
  isLoading: boolean;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
  updatePreferences: (preferences: Partial<UserPreferences>) => void;
  saveFaculty: (facultyId: string) => Promise<void>;
  unsaveFaculty: (facultyId: string) => Promise<void>;
  isFacultySaved: (facultyId: string) => boolean;
  addApplication: (application: Omit<Application, 'id' | 'userId' | 'createdAt' | 'updatedAt'>) => void;
  updateApplication: (id: string, updates: Partial<Application>) => void;
  deleteApplication: (id: string) => void;
  getApplicationByFacultyId: (facultyId: string) => Application | undefined;
}

const defaultPreferences: UserPreferences = {
  savedSearches: [],
  emailNotifications: true,
  weeklyDigest: true,
  newMatchAlerts: false,
  theme: 'auto',
};

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      savedFacultyIds: [],
      applications: [],
      isLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const { user, token } = await apiClient.login(email, password);

          // Load saved faculty
          const savedFacultyIds = await apiClient.getSavedFaculty();

          set({
            user: {
              ...user,
              preferences: user.preferences || defaultPreferences,
            },
            isAuthenticated: true,
            savedFacultyIds,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      register: async (email: string, password: string, name: string) => {
        set({ isLoading: true });
        try {
          const { user, token } = await apiClient.register(email, password, name);

          set({
            user: {
              ...user,
              preferences: defaultPreferences,
            },
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: () => {
        apiClient.logout();
        set({
          user: null,
          isAuthenticated: false,
          savedFacultyIds: [],
          applications: [],
        });
      },

      updatePreferences: (preferences: Partial<UserPreferences>) => {
        set((state) => ({
          user: state.user
            ? {
                ...state.user,
                preferences: {
                  ...state.user.preferences,
                  ...preferences,
                },
              }
            : null,
        }));
      },

      saveFaculty: async (facultyId: string) => {
        const state = get();
        if (state.savedFacultyIds.includes(facultyId)) {
          return;
        }

        try {
          await apiClient.saveFaculty(facultyId);
          set((state) => ({
            savedFacultyIds: [...state.savedFacultyIds, facultyId],
          }));
        } catch (error) {
          console.error('Failed to save faculty:', error);
          throw error;
        }
      },

      unsaveFaculty: async (facultyId: string) => {
        try {
          await apiClient.removeSavedFaculty(facultyId);
          set((state) => ({
            savedFacultyIds: state.savedFacultyIds.filter((id) => id !== facultyId),
          }));
        } catch (error) {
          console.error('Failed to unsave faculty:', error);
          throw error;
        }
      },

      isFacultySaved: (facultyId: string) => {
        return get().savedFacultyIds.includes(facultyId);
      },

      addApplication: (applicationData) => {
        const newApplication: Application = {
          ...applicationData,
          id: `app-${Date.now()}`,
          userId: get().user?.id || 'guest',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };

        set((state) => ({
          applications: [...state.applications, newApplication],
        }));
      },

      updateApplication: (id: string, updates: Partial<Application>) => {
        set((state) => ({
          applications: state.applications.map((app) =>
            app.id === id
              ? {
                  ...app,
                  ...updates,
                  updatedAt: new Date().toISOString(),
                }
              : app
          ),
        }));
      },

      deleteApplication: (id: string) => {
        set((state) => ({
          applications: state.applications.filter((app) => app.id !== id),
        }));
      },

      getApplicationByFacultyId: (facultyId: string) => {
        return get().applications.find((app) => app.facultyId === facultyId);
      },
    }),
    {
      name: 'user-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        savedFacultyIds: state.savedFacultyIds,
        applications: state.applications,
      }),
    }
  )
);
