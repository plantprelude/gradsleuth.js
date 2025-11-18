/**
 * API Client with automatic failover to mock data
 */

import axios, { AxiosInstance } from 'axios';
import type {
  Faculty,
  SearchQuery,
  SearchResult,
  Publication,
  Grant,
  MatchScore,
  SimilarityRequest,
  SimilarityResult,
  APIResponse,
} from '@/lib/types/api';
import { mockDataGenerator } from '@/services/mock/mockDataGenerator';

class APIClient {
  private client: AxiosInstance;
  private useMock: boolean;
  private apiUrl: string;

  constructor() {
    this.useMock = process.env.NEXT_PUBLIC_USE_MOCK === 'true';
    this.apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    this.client = axios.create({
      baseURL: this.apiUrl,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for adding auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for handling errors
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        // If API fails and not in mock mode, try to use mock data
        if (!this.useMock && error.code === 'ECONNREFUSED') {
          console.warn('API unavailable, falling back to mock data');
          this.useMock = true;
        }
        return Promise.reject(error);
      }
    );
  }

  private getAuthToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('auth_token');
    }
    return null;
  }

  private async simulateDelay() {
    // Simulate network delay for mock data
    if (this.useMock) {
      await new Promise((resolve) => setTimeout(resolve, 200 + Math.random() * 300));
    }
  }

  // ==================== Search & Discovery ====================

  async searchFaculty(query: SearchQuery): Promise<SearchResult> {
    if (this.useMock) {
      await this.simulateDelay();
      return mockDataGenerator.searchFaculty(query);
    }

    try {
      const response = await this.client.post<APIResponse<SearchResult>>(
        '/api/search',
        query
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Search API failed, using mock data', error);
      return mockDataGenerator.searchFaculty(query);
    }
  }

  async findSimilar(request: SimilarityRequest): Promise<SimilarityResult[]> {
    if (this.useMock) {
      await this.simulateDelay();
      // Generate mock similar results
      const allFaculty = mockDataGenerator.getFaculty(10);
      return allFaculty.slice(0, request.limit || 5).map((faculty) => ({
        faculty,
        score: Math.random() * 0.5 + 0.5, // 0.5-1.0
        matchScore: mockDataGenerator.generateMatchScore(
          faculty.id,
          request.text || ''
        ),
      }));
    }

    try {
      const response = await this.client.post<APIResponse<SimilarityResult[]>>(
        '/api/similar',
        request
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Similar API failed, using mock data', error);
      const allFaculty = mockDataGenerator.getFaculty(10);
      return allFaculty.slice(0, request.limit || 5).map((faculty) => ({
        faculty,
        score: Math.random() * 0.5 + 0.5,
        matchScore: mockDataGenerator.generateMatchScore(
          faculty.id,
          request.text || ''
        ),
      }));
    }
  }

  // ==================== Faculty ====================

  async getFaculty(id: string): Promise<Faculty | null> {
    if (this.useMock) {
      await this.simulateDelay();
      return mockDataGenerator.getFacultyById(id) || null;
    }

    try {
      const response = await this.client.get<APIResponse<Faculty>>(
        `/api/faculty/${id}`
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Get faculty API failed, using mock data', error);
      return mockDataGenerator.getFacultyById(id) || null;
    }
  }

  async getAllFaculty(limit: number = 50, offset: number = 0): Promise<Faculty[]> {
    if (this.useMock) {
      await this.simulateDelay();
      return mockDataGenerator.getFaculty(limit, offset);
    }

    try {
      const response = await this.client.get<APIResponse<Faculty[]>>(
        '/api/faculty',
        { params: { limit, offset } }
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Get all faculty API failed, using mock data', error);
      return mockDataGenerator.getFaculty(limit, offset);
    }
  }

  async getFacultyPublications(facultyId: string): Promise<Publication[]> {
    if (this.useMock) {
      await this.simulateDelay();
      const faculty = mockDataGenerator.getFacultyById(facultyId);
      if (!faculty) return [];
      return mockDataGenerator.generatePublications(
        facultyId,
        faculty.metrics.publicationCount
      );
    }

    try {
      const response = await this.client.get<APIResponse<Publication[]>>(
        `/api/faculty/${facultyId}/publications`
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Get publications API failed, using mock data', error);
      const faculty = mockDataGenerator.getFacultyById(facultyId);
      if (!faculty) return [];
      return mockDataGenerator.generatePublications(
        facultyId,
        faculty.metrics.publicationCount
      );
    }
  }

  async getFacultyGrants(facultyId: string): Promise<Grant[]> {
    if (this.useMock) {
      await this.simulateDelay();
      return mockDataGenerator.generateGrants(facultyId);
    }

    try {
      const response = await this.client.get<APIResponse<Grant[]>>(
        `/api/faculty/${facultyId}/grants`
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Get grants API failed, using mock data', error);
      return mockDataGenerator.generateGrants(facultyId);
    }
  }

  // ==================== Matching ====================

  async getMatchScore(facultyId: string, userQuery: string): Promise<MatchScore> {
    if (this.useMock) {
      await this.simulateDelay();
      return mockDataGenerator.generateMatchScore(facultyId, userQuery);
    }

    try {
      const response = await this.client.post<APIResponse<MatchScore>>(
        `/api/match/${facultyId}`,
        { query: userQuery }
      );
      return response.data.data!;
    } catch (error) {
      console.warn('Get match score API failed, using mock data', error);
      return mockDataGenerator.generateMatchScore(facultyId, userQuery);
    }
  }

  // ==================== User Management ====================

  async login(email: string, password: string): Promise<{ token: string; user: any }> {
    if (this.useMock) {
      await this.simulateDelay();
      // Mock login
      return {
        token: 'mock-jwt-token-' + Date.now(),
        user: {
          id: 'user-1',
          email,
          name: 'Demo User',
          role: 'student',
        },
      };
    }

    const response = await this.client.post<APIResponse<{ token: string; user: any }>>(
      '/api/auth/login',
      { email, password }
    );

    const data = response.data.data!;
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', data.token);
    }
    return data;
  }

  async register(email: string, password: string, name: string): Promise<{ token: string; user: any }> {
    if (this.useMock) {
      await this.simulateDelay();
      return {
        token: 'mock-jwt-token-' + Date.now(),
        user: {
          id: 'user-' + Date.now(),
          email,
          name,
          role: 'student',
        },
      };
    }

    const response = await this.client.post<APIResponse<{ token: string; user: any }>>(
      '/api/auth/register',
      { email, password, name }
    );

    const data = response.data.data!;
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', data.token);
    }
    return data;
  }

  logout(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  // ==================== Saved Data ====================

  async saveFaculty(facultyId: string): Promise<void> {
    if (this.useMock) {
      await this.simulateDelay();
      // Mock save to localStorage
      if (typeof window !== 'undefined') {
        const saved = JSON.parse(localStorage.getItem('saved_faculty') || '[]');
        if (!saved.includes(facultyId)) {
          saved.push(facultyId);
          localStorage.setItem('saved_faculty', JSON.stringify(saved));
        }
      }
      return;
    }

    await this.client.post('/api/user/saved-faculty', { facultyId });
  }

  async getSavedFaculty(): Promise<string[]> {
    if (this.useMock) {
      await this.simulateDelay();
      if (typeof window !== 'undefined') {
        return JSON.parse(localStorage.getItem('saved_faculty') || '[]');
      }
      return [];
    }

    const response = await this.client.get<APIResponse<string[]>>(
      '/api/user/saved-faculty'
    );
    return response.data.data!;
  }

  async removeSavedFaculty(facultyId: string): Promise<void> {
    if (this.useMock) {
      await this.simulateDelay();
      if (typeof window !== 'undefined') {
        const saved = JSON.parse(localStorage.getItem('saved_faculty') || '[]');
        const filtered = saved.filter((id: string) => id !== facultyId);
        localStorage.setItem('saved_faculty', JSON.stringify(filtered));
      }
      return;
    }

    await this.client.delete(`/api/user/saved-faculty/${facultyId}`);
  }

  // ==================== Health Check ====================

  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/api/health');
      return true;
    } catch {
      return false;
    }
  }

  // ==================== Mode Control ====================

  setMockMode(enabled: boolean): void {
    this.useMock = enabled;
  }

  isMockMode(): boolean {
    return this.useMock;
  }
}

// Export singleton instance
export const apiClient = new APIClient();
