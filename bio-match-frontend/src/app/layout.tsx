import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import '@/styles/globals.css';
import { Providers } from './providers';
import { Navigation } from '@/components/layout/Navigation';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'BioMatch - Biology Research Faculty Matching Platform',
  description:
    'Find the perfect faculty mentor for your graduate studies in biology. Search by research interests, techniques, and more.',
  keywords: [
    'biology',
    'graduate school',
    'faculty search',
    'research matching',
    'PhD',
    'mentorship',
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen flex flex-col">
            <Navigation />
            <main className="flex-1">{children}</main>
            <footer className="border-t py-6 px-4">
              <div className="container mx-auto text-center text-sm text-muted-foreground">
                <p>
                  © 2024 BioMatch. Data sourced from PubMed, NIH RePORTER, and
                  institutional databases.
                </p>
              </div>
            </footer>
          </div>
        </Providers>
      </body>
    </html>
  );
}
