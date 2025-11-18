# GradSleuth Architecture

This document outlines the current architecture of GradSleuth and the planned evolution as the platform grows.

## Table of Contents

- [System Overview](#system-overview)
- [Current Architecture (v0.0.0)](#current-architecture-v000)
- [Planned Architecture (v0.1.0 - MVP)](#planned-architecture-v010---mvp)
- [Future Architecture (v0.2.0+)](#future-architecture-v020)
- [Data Models](#data-models)
- [API Design](#api-design)
- [ML Pipeline](#ml-pipeline)
- [Infrastructure](#infrastructure)

---

## System Overview

GradSleuth is evolving from a simple frontend prototype into a full-stack research matching platform with ML capabilities.

### Core Capabilities

1. **Data Collection**: Aggregate faculty profiles, publications, grants
2. **Data Processing**: Clean, normalize, and enrich research data
3. **Semantic Understanding**: Extract research topics and expertise using NLP
4. **Matching**: Connect student interests with faculty research
5. **Presentation**: Display matches with relevance scores and insights

---

## Current Architecture (v0.0.0)

### System Diagram

```
┌─────────────┐
│   Browser   │
│             │
│ index.html  │
│ + Bootstrap │
│ + jQuery    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  gradsleuth.js  │
│                 │
│  - Topic input  │
│  - Query build  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  nickleby.js    │
│  (PubMed API)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  NCBI PubMed    │
│  E-utilities    │
└─────────────────┘
```

### Current Components

#### Frontend (`index.html`)
- **Purpose**: User interface for topic search
- **Tech**: HTML5, Bootstrap CSS, vanilla JavaScript
- **Features**:
  - Topic input form
  - Search button with Enter key support
  - Responsive navbar with logo

#### Core Logic (`assets/js/gradsleuth.js`)
- **Purpose**: Build PubMed search queries
- **Tech**: Node.js/CommonJS modules, nickleby.js
- **Features**:
  - Topic query construction
  - State-based affiliation filtering (all 50 US states)
  - PubMed search parameter building

#### Dependencies
- **nickleby.js**: Wrapper for NCBI E-utilities API
- **Bootstrap 4.x**: UI components and grid system
- **jQuery**: DOM manipulation and event handling

### Current Data Flow

```
User Input (Topic)
  → JavaScript handler (processTopic)
  → gradsleuth.js (query builder)
  → nickleby.js (API wrapper)
  → NCBI E-utilities
  → (Results to console - not yet displayed)
```

### Limitations

- No backend/database
- No persistent data storage
- No faculty profiles
- No actual search results displayed
- No ML/semantic matching
- Client-side only

---

## Planned Architecture (v0.1.0 - MVP)

### System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend (SPA)                        │
│  React + TypeScript + TailwindCSS                        │
│  - Search interface  - Faculty cards  - Filters          │
└─────────────────┬────────────────────────────────────────┘
                  │ REST API
                  ▼
┌──────────────────────────────────────────────────────────┐
│                  API Gateway (Node.js)                    │
│  Express + TypeScript                                    │
│  - Search endpoints  - Faculty endpoints  - Auth         │
└───┬──────────────┬──────────────┬───────────────────┬───┘
    │              │              │                   │
    ▼              ▼              ▼                   ▼
┌─────────┐  ┌──────────┐  ┌───────────┐      ┌──────────┐
│PostgreSQL│  │ Vector DB│  │  Cache    │      │  Queue   │
│         │  │ (pgvector│  │  (Redis)  │      │ (Bull)   │
│Faculty  │  │ /Pinecone│  │           │      │          │
│Pubs     │  │)         │  │Search     │      │Scraping  │
│Grants   │  │          │  │results    │      │jobs      │
└─────────┘  └──────────┘  └───────────┘      └────┬─────┘
                                                     │
                                                     ▼
                                              ┌────────────┐
                                              │  Workers   │
                                              │            │
                                              │ - Scrapers │
                                              │ - ETL      │
                                              │ - ML jobs  │
                                              └────────────┘
```

### Components

#### 1. Frontend (React SPA)
```
src/
├── components/
│   ├── Search/
│   │   ├── TopicInput.tsx
│   │   ├── Filters.tsx
│   │   └── SearchResults.tsx
│   ├── Faculty/
│   │   ├── FacultyCard.tsx
│   │   ├── FacultyProfile.tsx
│   │   └── PublicationList.tsx
│   └── Common/
│       ├── Header.tsx
│       └── Footer.tsx
├── hooks/
│   ├── useSearch.ts
│   └── useFaculty.ts
├── services/
│   └── api.ts
└── utils/
    └── formatting.ts
```

#### 2. API Gateway
```
src/
├── routes/
│   ├── search.ts
│   ├── faculty.ts
│   └── health.ts
├── services/
│   ├── searchService.ts
│   ├── facultyService.ts
│   └── embeddingService.ts
├── models/
│   ├── Faculty.ts
│   ├── Publication.ts
│   └── Grant.ts
└── middleware/
    ├── auth.ts
    ├── validation.ts
    └── errorHandler.ts
```

#### 3. Data Collection Workers
```
src/
├── scrapers/
│   ├── facultyProfileScraper.ts
│   ├── pubmedScraper.ts
│   └── nihReporterScraper.ts
├── processors/
│   ├── dataCleaner.ts
│   └── dataNormalizer.ts
└── jobs/
    ├── scheduledScrapes.ts
    └── embeddingGeneration.ts
```

---

## Future Architecture (v0.2.0+)

### Microservices Architecture

```
                    ┌──────────────┐
                    │  API Gateway │
                    │  (GraphQL)   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│ Search Service│  │Faculty Service│  │ ML Service   │
│               │  │              │  │              │
│ - Query parse │  │ - Profiles   │  │ - Embeddings │
│ - Ranking     │  │ - CRUD ops   │  │ - NER        │
│ - Filtering   │  │ - Enrichment │  │ - Clustering │
└───────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────┴───────┐
                    │  Event Bus   │
                    │  (RabbitMQ)  │
                    └──────────────┘
```

### Service Boundaries

#### Search Service
- Query parsing and intent detection
- Full-text search
- Vector similarity search
- Result ranking and aggregation
- Filtering and faceting

#### Faculty Service
- Profile CRUD operations
- Publication management
- Grant data management
- Collaboration network building
- Profile enrichment

#### ML Service
- Embedding generation (BioBERT, SciBERT)
- Named Entity Recognition (research topics, techniques)
- Text classification (research areas)
- Clustering (similar faculty)
- Recommendation engine

#### Data Collection Service
- Web scraping orchestration
- API integrations (PubMed, NIH)
- Data validation and cleaning
- Incremental updates
- Change detection

---

## Data Models

### Current (Conceptual)

No database yet, but planning for:

#### Faculty
```typescript
interface Faculty {
  id: string;
  name: string;
  institution: string;
  department: string;
  email?: string;
  website?: string;
  researchInterests: string[];
  bio: string;
  location: {
    city: string;
    state: string;
    country: string;
  };
  createdAt: Date;
  updatedAt: Date;
}
```

#### Publication
```typescript
interface Publication {
  id: string;
  pmid: string;  // PubMed ID
  title: string;
  abstract: string;
  authors: string[];
  journal: string;
  year: number;
  citationCount?: number;
  mesh_terms: string[];
  facultyId: string;
  embedding?: number[];  // 768-dim BioBERT vector
}
```

#### Grant
```typescript
interface Grant {
  id: string;
  nihProjectNumber: string;
  title: string;
  abstract: string;
  piName: string;
  institution: string;
  fiscalYear: number;
  awardAmount: number;
  agencyIC: string;  // NIH Institute/Center
  facultyId: string;
}
```

#### Search Query
```typescript
interface SearchQuery {
  id: string;
  userId?: string;
  query: string;
  filters: {
    states?: string[];
    institutions?: string[];
    yearRange?: [number, number];
  };
  embedding?: number[];
  timestamp: Date;
}
```

### Future: Vector Embeddings

```sql
-- Using pgvector extension
CREATE TABLE publication_embeddings (
  publication_id UUID PRIMARY KEY,
  embedding vector(768),  -- BioBERT embedding
  model_version VARCHAR(50),
  created_at TIMESTAMP
);

-- Vector similarity index
CREATE INDEX ON publication_embeddings
  USING ivfflat (embedding vector_cosine_ops);
```

---

## API Design

### Planned REST Endpoints (v0.1.0)

#### Search
```
GET  /api/v1/search
POST /api/v1/search/semantic
GET  /api/v1/search/suggestions
```

#### Faculty
```
GET    /api/v1/faculty
GET    /api/v1/faculty/:id
GET    /api/v1/faculty/:id/publications
GET    /api/v1/faculty/:id/grants
GET    /api/v1/faculty/:id/collaborators
```

#### Publications
```
GET /api/v1/publications
GET /api/v1/publications/:pmid
```

### Example Request/Response

**Request**: Semantic Search
```http
POST /api/v1/search/semantic
Content-Type: application/json

{
  "query": "CRISPR gene editing in cancer therapy",
  "filters": {
    "states": ["CA", "MA", "NY"],
    "yearRange": [2020, 2024]
  },
  "limit": 20
}
```

**Response**:
```json
{
  "results": [
    {
      "faculty": {
        "id": "fac_123",
        "name": "Dr. Jane Smith",
        "institution": "Stanford University",
        "department": "Genetics"
      },
      "matchScore": 0.94,
      "matchReasons": [
        "15 publications on CRISPR",
        "2 NIH grants on gene therapy",
        "Expertise in cancer genomics"
      ],
      "topPublications": [...]
    }
  ],
  "facets": {
    "institutions": [...],
    "researchAreas": [...]
  },
  "total": 47
}
```

---

## ML Pipeline

### Embedding Generation Pipeline

```
┌─────────────────┐
│  Raw Text       │
│  (Pub abstract) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Tokenization │
│  - Cleaning     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BioBERT Model  │
│  - Fine-tuned   │
│  - 768-dim out  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│  (pgvector)     │
└─────────────────┘
```

### Semantic Search Flow

```
User Query
  → Embed query with BioBERT
  → Vector similarity search (cosine)
  → Hybrid: combine with BM25 full-text
  → Re-rank with learning-to-rank model
  → Return top-K results
```

### Models to Use

1. **BioBERT**: Biomedical text embeddings
   - Base: `dmis-lab/biobert-v1.1`
   - Fine-tune on: PubMed abstracts + research interests

2. **SciBERT**: Scientific paper understanding
   - Use for: Broader scientific text

3. **NER**: Named Entity Recognition
   - SpaCy + custom biomedical NER
   - Extract: techniques, diseases, organisms, methods

4. **Clustering**: Group similar faculty
   - HDBSCAN on embeddings
   - Discover research communities

---

## Infrastructure

### Development (Current)
- Local development: Simple file serving
- No containerization yet
- Manual testing

### Staging (v0.1.0)
```yaml
Docker Compose:
  - Frontend container (Nginx + React build)
  - API container (Node.js)
  - PostgreSQL container
  - Redis container
  - Worker container
```

### Production (v0.2.0+)
```
Kubernetes Cluster:
  - Frontend: Static site (S3 + CloudFront or Vercel)
  - API: Auto-scaling pods
  - Database: Managed PostgreSQL (RDS/Cloud SQL)
  - Vector DB: Pinecone or self-hosted pgvector
  - ML Service: GPU instances for embedding
  - CDN: CloudFlare
  - Monitoring: Prometheus + Grafana
  - Logging: ELK stack
```

---

## Security Considerations

### Data Privacy
- No student PII stored without consent
- Faculty data from public sources only
- GDPR/CCPA compliance for user accounts

### API Security
- JWT authentication for user features
- Rate limiting on all endpoints
- Input validation and sanitization
- SQL injection prevention (parameterized queries)
- XSS protection

### Infrastructure Security
- HTTPS only
- Environment variable secrets management
- Database encryption at rest
- Network security groups
- Regular dependency updates

---

## Performance Targets

### v0.1.0 MVP
- Search latency: < 500ms (p95)
- Faculty profile load: < 200ms
- Concurrent users: 100
- Database size: 10K faculty, 500K publications

### v0.2.0 Production
- Search latency: < 200ms (p95)
- Semantic search: < 1s
- Concurrent users: 10,000
- Database size: 100K faculty, 5M publications
- Uptime: 99.9%

---

## Migration Path

### From v0.0.0 to v0.1.0

1. **Backend Setup**:
   - Initialize Express API
   - Set up PostgreSQL database
   - Create initial schema migrations

2. **Frontend Refactor**:
   - Convert to React/TypeScript
   - Keep existing UI design
   - Add API integration layer

3. **Data Pipeline**:
   - Build scrapers for faculty profiles
   - Import PubMed data
   - Create ETL jobs

4. **Testing**:
   - Add unit tests
   - Integration tests for API
   - E2E tests for critical flows

### From v0.1.0 to v0.2.0

1. **ML Integration**:
   - Train BioBERT embeddings
   - Generate embeddings for existing data
   - Deploy vector search

2. **Microservices Split**:
   - Extract ML service
   - Extract search service
   - Set up service mesh

3. **Scaling**:
   - Kubernetes deployment
   - Auto-scaling configuration
   - CDN setup

---

## Technology Decisions

### Why These Choices?

**PostgreSQL**:
- Relational data (faculty, pubs, grants) fits well
- pgvector extension for hybrid search
- Strong ACID guarantees

**Node.js/TypeScript**:
- JavaScript across stack
- Strong typing with TS
- Great async I/O for API
- Large ecosystem

**React**:
- Component-based UI
- Strong ecosystem
- Good TypeScript support
- Easy to find developers

**BioBERT**:
- State-of-art for biomedical text
- Pre-trained on PubMed
- Good embedding quality
- Reasonable inference speed

**Docker/Kubernetes**:
- Reproducible environments
- Easy scaling
- Industry standard
- Cloud-agnostic

---

## Future Considerations

- GraphQL as alternative to REST
- Server-side rendering for SEO
- Real-time features (WebSockets)
- Mobile app (React Native)
- Browser extension
- Slack/Discord integrations

---

**Document Version**: 1.0
**Last Updated**: November 2024
**Next Review**: After v0.1.0 MVP release
