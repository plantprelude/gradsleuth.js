/**
 * Mock Data Generator for Development
 * Generates realistic biology research data
 */

import { faker } from '@faker-js/faker';
import type {
  Faculty,
  Publication,
  Grant,
  SearchQuery,
  SearchResult,
  MatchScore,
} from '@/lib/types/api';

// Realistic biology research data
const UNIVERSITIES = [
  { name: 'Harvard University', id: 'harvard', city: 'Cambridge', state: 'MA' },
  { name: 'Stanford University', id: 'stanford', city: 'Stanford', state: 'CA' },
  { name: 'MIT', id: 'mit', city: 'Cambridge', state: 'MA' },
  { name: 'Yale University', id: 'yale', city: 'New Haven', state: 'CT' },
  { name: 'UC Berkeley', id: 'berkeley', city: 'Berkeley', state: 'CA' },
  { name: 'Johns Hopkins', id: 'jhu', city: 'Baltimore', state: 'MD' },
  { name: 'UCSF', id: 'ucsf', city: 'San Francisco', state: 'CA' },
  { name: 'Columbia University', id: 'columbia', city: 'New York', state: 'NY' },
  { name: 'UPenn', id: 'upenn', city: 'Philadelphia', state: 'PA' },
  { name: 'Duke University', id: 'duke', city: 'Durham', state: 'NC' },
];

const DEPARTMENTS = [
  'Biology',
  'Molecular Biology',
  'Cell Biology',
  'Genetics',
  'Neuroscience',
  'Biochemistry',
  'Immunology',
  'Microbiology',
  'Cancer Biology',
  'Developmental Biology',
];

const RESEARCH_AREAS = [
  'CRISPR gene editing',
  'Cancer immunotherapy',
  'Neurodegenerative diseases',
  'Stem cell biology',
  'Aging and longevity',
  'Synthetic biology',
  'Microbiome research',
  'Epigenetics',
  'RNA biology',
  'Signal transduction',
  'Developmental biology',
  'Systems biology',
  'Structural biology',
  'Computational biology',
  'Single-cell genomics',
];

const TECHNIQUES = [
  'CRISPR-Cas9',
  'RNA-seq',
  'Single-cell sequencing',
  'Cryo-EM',
  'Immunofluorescence',
  'Western blotting',
  'Flow cytometry',
  'ChIP-seq',
  'Mass spectrometry',
  'Optogenetics',
  'Two-photon microscopy',
  'Patch clamp',
  'X-ray crystallography',
  'FACS sorting',
  'Live cell imaging',
];

const ORGANISMS = [
  'Mouse',
  'Human',
  'Drosophila',
  'C. elegans',
  'Zebrafish',
  'Yeast',
  'E. coli',
  'Cell lines',
  'Organoids',
  'iPSCs',
];

const JOURNALS = [
  { name: 'Nature', if: 49.962 },
  { name: 'Science', if: 47.728 },
  { name: 'Cell', if: 64.5 },
  { name: 'Nature Genetics', if: 31.616 },
  { name: 'Nature Cell Biology', if: 28.824 },
  { name: 'Molecular Cell', if: 16.0 },
  { name: 'Cell Stem Cell', if: 23.9 },
  { name: 'Neuron', if: 17.173 },
  { name: 'PLOS Biology', if: 9.163 },
  { name: 'eLife', if: 8.713 },
];

const GRANT_MECHANISMS = [
  'R01', 'R21', 'R35', 'K99', 'K08', 'R00', 'DP2', 'U01', 'P01', 'R37'
];

const NIH_INSTITUTES = [
  'NIGMS', 'NCI', 'NIMH', 'NINDS', 'NIAID', 'NHLBI', 'NICHD', 'NEI', 'NIA'
];

class MockDataGenerator {
  private facultyCache: Faculty[] = [];
  private publicationCache: Map<string, Publication[]> = new Map();
  private grantCache: Map<string, Grant[]> = new Map();

  constructor() {
    this.initializeCache();
  }

  private initializeCache() {
    // Generate initial set of faculty
    this.facultyCache = Array.from({ length: 500 }, (_, i) =>
      this.generateSingleFaculty(i)
    );
  }

  private generateSingleFaculty(index: number): Faculty {
    const university = faker.helpers.arrayElement(UNIVERSITIES);
    const department = faker.helpers.arrayElement(DEPARTMENTS);
    const careerStage = faker.helpers.arrayElement(['early', 'mid', 'senior'] as const);
    const yearsActive = careerStage === 'early' ? faker.number.int({ min: 1, max: 7 }) :
                        careerStage === 'mid' ? faker.number.int({ min: 8, max: 15 }) :
                        faker.number.int({ min: 16, max: 40 });

    const firstName = faker.person.firstName();
    const lastName = faker.person.lastName();
    const name = `${firstName} ${lastName}`;

    const publicationCount = Math.floor(yearsActive * faker.number.float({ min: 2, max: 8 }));
    const avgCitations = faker.number.int({ min: 10, max: 100 });
    const totalCitations = Math.floor(publicationCount * avgCitations * faker.number.float({ min: 0.5, max: 1.5 }));
    const hIndex = Math.floor(Math.sqrt(totalCitations / 10));

    const techniques = faker.helpers.arrayElements(TECHNIQUES, faker.number.int({ min: 3, max: 8 }));
    const organisms = faker.helpers.arrayElements(ORGANISMS, faker.number.int({ min: 1, max: 4 }));
    const researchAreas = faker.helpers.arrayElements(RESEARCH_AREAS, faker.number.int({ min: 2, max: 5 }));

    const faculty: Faculty = {
      id: `faculty-${index}`,
      personalInfo: {
        name,
        firstName,
        lastName,
        title: careerStage === 'senior' ? 'Professor' : careerStage === 'mid' ? 'Associate Professor' : 'Assistant Professor',
        email: `${firstName.toLowerCase()}.${lastName.toLowerCase()}@${university.id}.edu`,
        phone: faker.phone.number(),
        office: `${faker.location.buildingNumber()} ${faker.helpers.arrayElement(['Main', 'Science', 'Research'])} Building`,
        photoUrl: `https://i.pravatar.cc/300?img=${index}`,
        orcid: faker.string.numeric(16),
        googleScholarId: faker.string.alphanumeric(12),
      },
      affiliation: {
        university: university.name,
        universityId: university.id,
        department,
        departmentId: department.toLowerCase().replace(/\s+/g, '-'),
        programAffiliations: faker.helpers.arrayElements(
          ['Graduate Program in Biology', 'Cancer Biology Program', 'Neuroscience Program'],
          faker.number.int({ min: 1, max: 3 })
        ),
        location: {
          city: university.city,
          state: university.state,
          country: 'USA',
          latitude: faker.location.latitude(),
          longitude: faker.location.longitude(),
        },
      },
      research: {
        interests: researchAreas,
        summary: this.generateResearchSummary(researchAreas, techniques, organisms),
        keywords: faker.helpers.arrayElements(
          [...researchAreas, ...techniques.slice(0, 3)],
          faker.number.int({ min: 5, max: 10 })
        ),
        organisms,
        techniques,
        meshTerms: faker.helpers.arrayElements(researchAreas, faker.number.int({ min: 3, max: 7 })),
        recentFocus: faker.helpers.arrayElements(researchAreas, faker.number.int({ min: 1, max: 3 })),
        researchAreas: researchAreas.map((area, i) => ({
          name: area,
          level: i === 0 ? 'primary' : i === 1 ? 'secondary' : 'tertiary',
          weight: i === 0 ? 1.0 : i === 1 ? 0.6 : 0.3,
        })),
      },
      lab: {
        name: `${lastName} Lab`,
        website: `https://www.${university.id}.edu/${lastName.toLowerCase()}lab`,
        size: faker.number.int({ min: 3, max: 20 }),
        gradStudents: faker.number.int({ min: 1, max: 6 }),
        postdocs: faker.number.int({ min: 0, max: 5 }),
        acceptingStudents: faker.datatype.boolean(0.7),
        rotationAvailability: faker.helpers.arrayElement(['available', 'maybe', 'full'] as const),
        labCulture: faker.helpers.arrayElements(
          ['Collaborative', 'Independent', 'Mentorship-focused', 'High-impact', 'Work-life balance'],
          faker.number.int({ min: 2, max: 4 })
        ),
      },
      metrics: {
        publicationCount,
        hIndex,
        i10Index: Math.floor(hIndex * 0.8),
        totalCitations,
        recentPublications: Math.floor(publicationCount * 0.15),
        avgCitationsPerPaper: Math.floor(totalCitations / publicationCount),
        activeFunding: faker.number.float({ min: 0.5, max: 8, precision: 0.1 }),
        totalFundingHistory: faker.number.float({ min: 2, max: 50, precision: 0.1 }),
        activeGrants: faker.number.int({ min: 1, max: 4 }),
        careerStage,
        yearsActive,
        firstPublicationYear: new Date().getFullYear() - yearsActive,
      },
      embedding: this.generateMockEmbedding(researchAreas.join(' ') + ' ' + techniques.join(' ')),
      lastUpdated: faker.date.recent({ days: 30 }).toISOString(),
    };

    return faculty;
  }

  private generateResearchSummary(areas: string[], techniques: string[], organisms: string[]): string {
    const summaries = [
      `Our lab studies ${areas[0].toLowerCase()} using ${techniques[0]} in ${organisms[0]} models. We are particularly interested in understanding the molecular mechanisms underlying ${areas[1]?.toLowerCase() || 'disease progression'}.`,
      `We use ${techniques[0]} and ${techniques[1]?.toLowerCase() || 'other approaches'} to investigate ${areas[0].toLowerCase()}. Our work in ${organisms[0]} has revealed new insights into ${areas[1]?.toLowerCase() || 'cellular function'}.`,
      `Research in our lab focuses on ${areas[0].toLowerCase()} with applications to ${areas[1]?.toLowerCase() || 'human disease'}. We employ ${techniques[0]} to study ${organisms[0]} and translate findings to clinical contexts.`,
    ];
    return faker.helpers.arrayElement(summaries);
  }

  private generateMockEmbedding(text: string): number[] {
    // Generate deterministic embedding based on text
    // In real app, this would come from BioBERT
    const seed = text.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    faker.seed(seed);
    return Array.from({ length: 768 }, () => faker.number.float({ min: -1, max: 1 }));
  }

  generatePublications(facultyId: string, count: number): Publication[] {
    if (this.publicationCache.has(facultyId)) {
      return this.publicationCache.get(facultyId)!;
    }

    const faculty = this.facultyCache.find(f => f.id === facultyId);
    if (!faculty) return [];

    const currentYear = new Date().getFullYear();
    const publications = Array.from({ length: count }, (_, i) => {
      const year = currentYear - faker.number.int({ min: 0, max: faculty.metrics.yearsActive });
      const journal = faker.helpers.arrayElement(JOURNALS);
      const citations = Math.floor(
        faker.number.exponential(1 / 50) * (currentYear - year + 1)
      );

      const authorCount = faker.number.int({ min: 3, max: 12 });
      const facultyPosition = faker.number.int({ min: 0, max: authorCount - 1 });

      const publication: Publication = {
        id: `pub-${facultyId}-${i}`,
        pmid: faker.string.numeric(8),
        doi: `10.1038/${faker.string.alphanumeric(8)}`,
        title: this.generatePublicationTitle(faculty.research.interests),
        abstract: this.generateAbstract(faculty.research.interests, faculty.research.techniques),
        authors: Array.from({ length: authorCount }, (_, j) => ({
          name: faker.person.fullName(),
          firstName: faker.person.firstName(),
          lastName: faker.person.lastName(),
          affiliation: faker.helpers.arrayElement(UNIVERSITIES).name,
          isCorresponding: j === authorCount - 1,
        })),
        journal: journal.name,
        journalImpactFactor: journal.if,
        year,
        month: faker.number.int({ min: 1, max: 12 }),
        volume: faker.string.numeric(2),
        issue: faker.string.numeric(1),
        pages: `${faker.number.int({ min: 100, max: 999 })}-${faker.number.int({ min: 100, max: 999 })}`,
        citations,
        keywords: faker.helpers.arrayElements(faculty.research.keywords, faker.number.int({ min: 3, max: 8 })),
        meshTerms: faker.helpers.arrayElements(faculty.research.meshTerms, faker.number.int({ min: 3, max: 6 })),
        publicationType: ['Journal Article', 'Research Support, N.I.H., Extramural'],
        facultyId,
        facultyAuthorPosition: facultyPosition,
        isFirstAuthor: facultyPosition === 0,
        isLastAuthor: facultyPosition === authorCount - 1,
        isCorrespondingAuthor: facultyPosition === authorCount - 1,
        embedding: this.generateMockEmbedding(faculty.research.interests.join(' ')),
      };

      return publication;
    });

    this.publicationCache.set(facultyId, publications);
    return publications;
  }

  private generatePublicationTitle(interests: string[]): string {
    const titles = [
      `${interests[0]} regulates ${interests[1]?.toLowerCase() || 'cellular function'} through novel signaling pathways`,
      `Molecular mechanisms of ${interests[0].toLowerCase()} in disease progression`,
      `A comprehensive analysis of ${interests[0].toLowerCase()} using multi-omics approaches`,
      `${interests[0]} controls ${interests[1]?.toLowerCase() || 'gene expression'} during development`,
      `Novel insights into ${interests[0].toLowerCase()}: implications for therapeutic targeting`,
    ];
    return faker.helpers.arrayElement(titles);
  }

  private generateAbstract(interests: string[], techniques: string[]): string {
    return `Background: ${interests[0]} plays a critical role in cellular function. Methods: We used ${techniques[0]} and ${techniques[1]?.toLowerCase() || 'other techniques'} to investigate the molecular mechanisms. Results: Our findings reveal novel pathways regulating ${interests[1]?.toLowerCase() || 'cellular processes'}. Conclusions: These results have important implications for understanding ${interests[0].toLowerCase()} and potential therapeutic applications.`;
  }

  generateGrants(facultyId: string): Grant[] {
    if (this.grantCache.has(facultyId)) {
      return this.grantCache.get(facultyId)!;
    }

    const faculty = this.facultyCache.find(f => f.id === facultyId);
    if (!faculty) return [];

    const grantCount = faculty.metrics.activeGrants + faker.number.int({ min: 1, max: 3 });
    const currentYear = new Date().getFullYear();

    const grants = Array.from({ length: grantCount }, (_, i) => {
      const mechanism = faker.helpers.arrayElement(GRANT_MECHANISMS);
      const startYear = currentYear - faker.number.int({ min: 0, max: 5 });
      const duration = mechanism.startsWith('R01') || mechanism === 'R35' ? 5 :
                      mechanism.startsWith('R21') ? 2 : 3;
      const endYear = startYear + duration;
      const isActive = endYear >= currentYear && startYear <= currentYear;

      const directCosts = mechanism.startsWith('R01') ?
        faker.number.float({ min: 200000, max: 350000, precision: 1000 }) :
        mechanism.startsWith('R21') ?
        faker.number.float({ min: 100000, max: 200000, precision: 1000 }) :
        faker.number.float({ min: 150000, max: 400000, precision: 1000 });

      const totalAmount = directCosts * duration * 1.5; // Include indirect costs

      const grant: Grant = {
        id: `grant-${facultyId}-${i}`,
        projectNumber: `${mechanism}${faker.string.alphanumeric(6).toUpperCase()}`,
        title: `Investigating ${faculty.research.interests[0]} mechanisms in ${faker.helpers.arrayElement(faculty.research.organisms)} models`,
        abstract: this.generateAbstract(faculty.research.interests, faculty.research.techniques),
        totalAmount,
        directCosts,
        indirectCosts: totalAmount - directCosts,
        startDate: `${startYear}-01-01`,
        endDate: `${endYear}-12-31`,
        agency: 'NIH',
        institute: faker.helpers.arrayElement(NIH_INSTITUTES),
        mechanism,
        activityCode: mechanism,
        piId: facultyId,
        piName: faculty.personalInfo.name,
        coPIs: faker.helpers.multiple(() => ({
          name: faker.person.fullName(),
          facultyId: faker.datatype.boolean(0.3) ? faker.helpers.arrayElement(this.facultyCache).id : undefined,
        }), { count: faker.number.int({ min: 0, max: 2 }) }),
        isActive,
        organization: faculty.affiliation.university,
        keywords: faculty.research.keywords.slice(0, 5),
      };

      return grant;
    });

    this.grantCache.set(facultyId, grants);
    return grants;
  }

  getFaculty(limit: number = 50, offset: number = 0): Faculty[] {
    return this.facultyCache.slice(offset, offset + limit);
  }

  getFacultyById(id: string): Faculty | undefined {
    return this.facultyCache.find(f => f.id === id);
  }

  searchFaculty(query: SearchQuery): SearchResult {
    const startTime = Date.now();
    let results = [...this.facultyCache];

    // Apply text search
    if (query.query) {
      const searchLower = query.query.toLowerCase();
      results = results.filter(f =>
        f.personalInfo.name.toLowerCase().includes(searchLower) ||
        f.research.interests.some(i => i.toLowerCase().includes(searchLower)) ||
        f.research.keywords.some(k => k.toLowerCase().includes(searchLower)) ||
        f.research.techniques.some(t => t.toLowerCase().includes(searchLower)) ||
        f.research.summary.toLowerCase().includes(searchLower)
      );
    }

    // Apply filters
    if (query.filters.universities?.length) {
      results = results.filter(f =>
        query.filters.universities!.includes(f.affiliation.university)
      );
    }

    if (query.filters.departments?.length) {
      results = results.filter(f =>
        query.filters.departments!.includes(f.affiliation.department)
      );
    }

    if (query.filters.states?.length) {
      results = results.filter(f =>
        query.filters.states!.includes(f.affiliation.location.state)
      );
    }

    if (query.filters.techniques?.length) {
      results = results.filter(f =>
        query.filters.techniques!.some(t => f.research.techniques.includes(t))
      );
    }

    if (query.filters.organisms?.length) {
      results = results.filter(f =>
        query.filters.organisms!.some(o => f.research.organisms.includes(o))
      );
    }

    if (query.filters.acceptingStudents !== undefined) {
      results = results.filter(f => f.lab.acceptingStudents === query.filters.acceptingStudents);
    }

    if (query.filters.fundingMin !== undefined) {
      results = results.filter(f => f.metrics.activeFunding >= query.filters.fundingMin!);
    }

    if (query.filters.careerStage?.length) {
      results = results.filter(f => query.filters.careerStage!.includes(f.metrics.careerStage));
    }

    // Apply sorting
    switch (query.sort) {
      case 'funding-desc':
        results.sort((a, b) => b.metrics.activeFunding - a.metrics.activeFunding);
        break;
      case 'publications-desc':
        results.sort((a, b) => b.metrics.publicationCount - a.metrics.publicationCount);
        break;
      case 'citations-desc':
        results.sort((a, b) => b.metrics.totalCitations - a.metrics.totalCitations);
        break;
      case 'h-index-desc':
        results.sort((a, b) => b.metrics.hIndex - a.metrics.hIndex);
        break;
      case 'name-asc':
        results.sort((a, b) => a.personalInfo.name.localeCompare(b.personalInfo.name));
        break;
    }

    // Generate facets
    const facets = this.generateFacets(results);

    // Pagination
    const page = query.page || 1;
    const limit = query.limit || 20;
    const paginatedResults = results.slice((page - 1) * limit, page * limit);

    const queryTime = Date.now() - startTime;

    return {
      faculty: paginatedResults,
      totalCount: results.length,
      page,
      limit,
      facets,
      queryTime,
    };
  }

  private generateFacets(results: Faculty[]) {
    const countMap = <T>(items: T[]) => {
      const counts = new Map<T, number>();
      items.forEach(item => counts.set(item, (counts.get(item) || 0) + 1));
      return Array.from(counts.entries())
        .map(([value, count]) => ({ value: String(value), count }))
        .sort((a, b) => b.count - a.count);
    };

    return {
      universities: countMap(results.map(f => f.affiliation.university)),
      departments: countMap(results.map(f => f.affiliation.department)),
      states: countMap(results.map(f => f.affiliation.location.state)),
      techniques: countMap(results.flatMap(f => f.research.techniques)),
      organisms: countMap(results.flatMap(f => f.research.organisms)),
      keywords: countMap(results.flatMap(f => f.research.keywords)).slice(0, 20),
      researchAreas: countMap(results.flatMap(f => f.research.interests)),
      careerStages: countMap(results.map(f => f.metrics.careerStage)),
      grantTypes: [
        { value: 'R01', count: faker.number.int({ min: 50, max: 200 }) },
        { value: 'R21', count: faker.number.int({ min: 20, max: 100 }) },
        { value: 'K99', count: faker.number.int({ min: 10, max: 50 }) },
      ],
    };
  }

  generateMatchScore(facultyId: string, userQuery: string): MatchScore {
    const faculty = this.getFacultyById(facultyId);
    if (!faculty) {
      throw new Error('Faculty not found');
    }

    // Simple mock scoring based on keyword matching
    const queryLower = userQuery.toLowerCase();
    const researchMatch = faculty.research.interests.some(i =>
      queryLower.includes(i.toLowerCase())
    ) || faculty.research.keywords.some(k =>
      queryLower.includes(k.toLowerCase())
    );

    const breakdown = {
      researchAlignment: faker.number.int({ min: researchMatch ? 70 : 40, max: 95 }),
      fundingStrength: Math.min(95, faculty.metrics.activeFunding * 15),
      publicationImpact: Math.min(95, faculty.metrics.hIndex * 2),
      labCulture: faker.number.int({ min: 60, max: 90 }),
      careerStage: faculty.metrics.careerStage === 'early' ? 85 :
                   faculty.metrics.careerStage === 'mid' ? 75 : 65,
      availability: faculty.lab.acceptingStudents ? 90 : 40,
    };

    const overallScore = Math.round(
      Object.values(breakdown).reduce((a, b) => a + b, 0) / 6
    );

    return {
      facultyId,
      overallScore,
      breakdown,
      explanation: [
        `Strong match in ${faculty.research.interests[0]}`,
        `Currently accepting students with ${faculty.lab.size} lab members`,
        `Active funding of $${faculty.metrics.activeFunding.toFixed(1)}M`,
        `H-index of ${faculty.metrics.hIndex} with ${faculty.metrics.publicationCount} publications`,
      ],
      confidence: faker.number.float({ min: 0.7, max: 0.95, precision: 0.01 }),
    };
  }
}

// Export singleton instance
export const mockDataGenerator = new MockDataGenerator();
