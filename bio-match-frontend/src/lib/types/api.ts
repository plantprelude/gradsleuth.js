/**
 * API Type Definitions for Biology Research Matching Platform
 * These interfaces define the contract between frontend and backend
 */

// ==================== Core Faculty Types ====================

export interface Faculty {
  id: string;
  personalInfo: PersonalInfo;
  affiliation: Affiliation;
  research: Research;
  lab: LabInfo;
  metrics: Metrics;
  embedding?: number[]; // BioBERT embeddings for similarity
  lastUpdated: string;
}

export interface PersonalInfo {
  name: string;
  firstName: string;
  lastName: string;
  title: string;
  email: string;
  phone?: string;
  office?: string;
  photoUrl?: string;
  personalWebsite?: string;
  orcid?: string;
  googleScholarId?: string;
}

export interface Affiliation {
  university: string;
  universityId: string;
  department: string;
  departmentId: string;
  division?: string;
  programAffiliations: string[];
  location: Location;
}

export interface Location {
  city: string;
  state: string;
  country: string;
  latitude?: number;
  longitude?: number;
}

export interface Research {
  interests: string[];
  summary: string;
  keywords: string[];
  organisms: string[];
  techniques: string[];
  meshTerms: string[];
  recentFocus: string[]; // Topics from last 12 months
  researchAreas: ResearchArea[];
}

export interface ResearchArea {
  name: string;
  level: 'primary' | 'secondary' | 'tertiary';
  weight: number; // 0-1 indicating importance
}

export interface LabInfo {
  name?: string;
  website?: string;
  size: number; // Number of lab members
  gradStudents: number;
  postdocs: number;
  acceptingStudents: boolean;
  rotationAvailability: 'available' | 'maybe' | 'full';
  labCulture?: string[];
  expectations?: string;
}

export interface Metrics {
  publicationCount: number;
  hIndex: number;
  i10Index: number;
  totalCitations: number;
  recentPublications: number; // Last 2 years
  avgCitationsPerPaper: number;
  activeFunding: number; // In millions USD
  totalFundingHistory: number;
  activeGrants: number;
  careerStage: 'early' | 'mid' | 'senior' | 'emeritus';
  yearsActive: number;
  firstPublicationYear?: number;
}

// ==================== Publication Types ====================

export interface Publication {
  id: string;
  pmid: string;
  doi?: string;
  title: string;
  abstract: string;
  authors: Author[];
  journal: string;
  journalImpactFactor?: number;
  year: number;
  month?: number;
  volume?: string;
  issue?: string;
  pages?: string;
  citations: number;
  keywords: string[];
  meshTerms: string[];
  publicationType: string[];
  facultyId: string;
  facultyAuthorPosition: number; // 0-indexed position in author list
  isFirstAuthor: boolean;
  isLastAuthor: boolean;
  isCorrespondingAuthor: boolean;
  embedding?: number[];
}

export interface Author {
  name: string;
  firstName?: string;
  lastName?: string;
  affiliation?: string;
  orcid?: string;
  isCorresponding?: boolean;
}

// ==================== Grant Types ====================

export interface Grant {
  id: string;
  projectNumber: string;
  title: string;
  abstract: string;
  totalAmount: number;
  directCosts: number;
  indirectCosts: number;
  startDate: string;
  endDate: string;
  agency: string;
  institute?: string; // e.g., NIGMS, NCI for NIH
  mechanism: string; // R01, R21, K99, etc.
  activityCode: string;
  piId: string;
  piName: string;
  coPIs: CoPI[];
  isActive: boolean;
  organization: string;
  keywords: string[];
}

export interface CoPI {
  name: string;
  facultyId?: string;
}

// ==================== Search Types ====================

export interface SearchQuery {
  query: string;
  filters: SearchFilters;
  sort?: SortOption;
  page?: number;
  limit?: number;
  includeEmbeddings?: boolean;
}

export interface SearchFilters {
  universities?: string[];
  departments?: string[];
  states?: string[];
  countries?: string[];
  fundingMin?: number;
  fundingMax?: number;
  publicationMin?: number;
  hIndexMin?: number;
  techniques?: string[];
  organisms?: string[];
  researchAreas?: string[];
  keywords?: string[];
  acceptingStudents?: boolean;
  rotationAvailable?: boolean;
  careerStage?: ('early' | 'mid' | 'senior')[];
  grantTypes?: string[];
  location?: LocationFilter;
  yearsActiveMin?: number;
  yearsActiveMax?: number;
}

export interface LocationFilter {
  latitude: number;
  longitude: number;
  radiusMiles: number;
}

export type SortOption =
  | 'relevance'
  | 'funding-desc'
  | 'funding-asc'
  | 'publications-desc'
  | 'publications-asc'
  | 'citations-desc'
  | 'h-index-desc'
  | 'recent-activity'
  | 'name-asc'
  | 'name-desc';

export interface SearchResult {
  faculty: Faculty[];
  totalCount: number;
  page: number;
  limit: number;
  facets: SearchFacets;
  queryTime: number; // milliseconds
  suggestedQuery?: string;
}

export interface SearchFacets {
  universities: FacetValue[];
  departments: FacetValue[];
  states: FacetValue[];
  techniques: FacetValue[];
  organisms: FacetValue[];
  keywords: FacetValue[];
  researchAreas: FacetValue[];
  careerStages: FacetValue[];
  grantTypes: FacetValue[];
}

export interface FacetValue {
  value: string;
  count: number;
  selected?: boolean;
}

// ==================== Matching & Scoring Types ====================

export interface MatchScore {
  facultyId: string;
  overallScore: number; // 0-100
  breakdown: ScoreBreakdown;
  explanation: string[];
  similarityVector?: number[];
  confidence: number; // 0-1
}

export interface ScoreBreakdown {
  researchAlignment: number; // 0-100
  fundingStrength: number;
  publicationImpact: number;
  labCulture: number;
  careerStage: number;
  availability: number;
}

export interface SimilarityRequest {
  text?: string;
  facultyId?: string;
  publicationId?: string;
  embedding?: number[];
  limit?: number;
  minScore?: number;
}

export interface SimilarityResult {
  faculty: Faculty;
  score: number; // 0-1 cosine similarity
  matchScore: MatchScore;
}

// ==================== User & Application Types ====================

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'student' | 'faculty' | 'admin';
  profile?: StudentProfile;
  preferences: UserPreferences;
  createdAt: string;
  lastLogin: string;
}

export interface StudentProfile {
  institution?: string;
  expectedGradYear?: number;
  researchInterests: string[];
  techniques: string[];
  careerGoals?: string;
  gpa?: number;
  gre?: GREScores;
  publications?: string[]; // Publication IDs
  cvUrl?: string;
}

export interface GREScores {
  verbal?: number;
  quantitative?: number;
  analytical?: number;
}

export interface UserPreferences {
  savedSearches: SavedSearch[];
  emailNotifications: boolean;
  weeklyDigest: boolean;
  newMatchAlerts: boolean;
  theme: 'light' | 'dark' | 'auto';
}

export interface SavedSearch {
  id: string;
  name: string;
  query: SearchQuery;
  createdAt: string;
  lastRun?: string;
  alertEnabled: boolean;
  resultCount?: number;
}

export interface Application {
  id: string;
  userId: string;
  facultyId: string;
  status: ApplicationStatus;
  notes: string;
  documents: Document[];
  timeline: TimelineEvent[];
  deadlines: Deadline[];
  createdAt: string;
  updatedAt: string;
}

export type ApplicationStatus =
  | 'researching'
  | 'contacting'
  | 'preparing'
  | 'submitted'
  | 'interviewing'
  | 'accepted'
  | 'rejected'
  | 'withdrawn';

export interface Document {
  id: string;
  name: string;
  type: 'cv' | 'cover-letter' | 'transcript' | 'other';
  url: string;
  uploadedAt: string;
}

export interface TimelineEvent {
  id: string;
  date: string;
  type: string;
  description: string;
  createdAt: string;
}

export interface Deadline {
  id: string;
  name: string;
  date: string;
  completed: boolean;
  priority: 'high' | 'medium' | 'low';
}

// ==================== Comparison Types ====================

export interface ComparisonData {
  faculty: Faculty[];
  metrics: ComparisonMetrics;
  charts: ComparisonChartData[];
}

export interface ComparisonMetrics {
  [facultyId: string]: {
    strengths: string[];
    weaknesses: string[];
    unique: string[];
  };
}

export interface ComparisonChartData {
  type: 'radar' | 'bar' | 'line';
  data: any;
  title: string;
}

// ==================== Analytics Types ====================

export interface Analytics {
  pageViews: PageViewData[];
  searchAnalytics: SearchAnalytics;
  userBehavior: UserBehavior;
  conversionMetrics: ConversionMetrics;
}

export interface PageViewData {
  path: string;
  count: number;
  avgDuration: number;
}

export interface SearchAnalytics {
  topQueries: QueryCount[];
  topFilters: FilterUsage[];
  avgResultsCount: number;
  avgQueryTime: number;
}

export interface QueryCount {
  query: string;
  count: number;
}

export interface FilterUsage {
  filter: string;
  count: number;
}

export interface UserBehavior {
  avgSessionDuration: number;
  avgPagesPerSession: number;
  bounceRate: number;
  returnUserRate: number;
}

export interface ConversionMetrics {
  signupRate: number;
  saveRate: number;
  applicationRate: number;
  contactRate: number;
}

// ==================== API Response Types ====================

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: APIError;
  meta?: ResponseMeta;
}

export interface APIError {
  code: string;
  message: string;
  details?: any;
}

export interface ResponseMeta {
  timestamp: string;
  requestId: string;
  version: string;
}

// ==================== Visualization Data Types ====================

export interface NetworkNode {
  id: string;
  name: string;
  type: 'faculty' | 'publication' | 'grant' | 'keyword';
  value: number;
  group?: string;
}

export interface NetworkLink {
  source: string;
  target: string;
  value: number;
  type: string;
}

export interface NetworkData {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

export interface TimeSeriesData {
  date: string;
  value: number;
  label?: string;
}

export interface WordCloudData {
  text: string;
  value: number;
  color?: string;
}
