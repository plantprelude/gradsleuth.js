'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils/cn';
import {
  Search,
  Home,
  User,
  BookmarkIcon,
  GitCompare,
  Menu,
  X,
} from 'lucide-react';
import { useUserStore } from '@/lib/stores/userStore';
import { useComparisonStore } from '@/lib/stores/comparisonStore';
import { Badge } from '@/components/ui/badge';

export function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);
  const pathname = usePathname();
  const isAuthenticated = useUserStore((state) => state.isAuthenticated);
  const user = useUserStore((state) => state.user);
  const logout = useUserStore((state) => state.logout);
  const comparisonCount = useComparisonStore(
    (state) => state.selectedFacultyIds.length
  );

  const navLinks = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/search', label: 'Search', icon: Search },
    ...(isAuthenticated
      ? [
          { href: '/dashboard', label: 'Dashboard', icon: User },
          { href: '/dashboard/saved', label: 'Saved', icon: BookmarkIcon },
        ]
      : []),
  ];

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center px-4">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2 mr-6">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xl">B</span>
          </div>
          <span className="font-bold text-xl">BioMatch</span>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-6 flex-1">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  'flex items-center space-x-2 text-sm font-medium transition-colors hover:text-primary',
                  pathname === link.href
                    ? 'text-primary'
                    : 'text-muted-foreground'
                )}
              >
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center space-x-4">
          {/* Comparison Badge */}
          {comparisonCount > 0 && (
            <Link href="/faculty/compare">
              <Button variant="outline" size="sm" className="relative">
                <GitCompare className="w-4 h-4 mr-2" />
                Compare
                <Badge
                  variant="destructive"
                  className="ml-2 px-1.5 py-0 text-xs"
                >
                  {comparisonCount}
                </Badge>
              </Button>
            </Link>
          )}

          {/* Auth */}
          {isAuthenticated ? (
            <div className="hidden md:flex items-center space-x-2">
              <span className="text-sm text-muted-foreground">
                {user?.name}
              </span>
              <Button variant="ghost" size="sm" onClick={logout}>
                Logout
              </Button>
            </div>
          ) : (
            <div className="hidden md:flex items-center space-x-2">
              <Link href="/auth/login">
                <Button variant="ghost" size="sm">
                  Login
                </Button>
              </Link>
              <Link href="/auth/register">
                <Button size="sm">Sign Up</Button>
              </Link>
            </div>
          )}

          {/* Mobile Menu Toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </Button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t bg-background">
          <div className="container mx-auto px-4 py-4 space-y-3">
            {navLinks.map((link) => {
              const Icon = link.icon;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cn(
                    'flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    pathname === link.href
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-accent'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span>{link.label}</span>
                </Link>
              );
            })}

            <div className="border-t pt-3 mt-3">
              {isAuthenticated ? (
                <>
                  <div className="px-3 py-2 text-sm text-muted-foreground">
                    {user?.name}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start"
                    onClick={() => {
                      logout();
                      setMobileMenuOpen(false);
                    }}
                  >
                    Logout
                  </Button>
                </>
              ) : (
                <div className="space-y-2">
                  <Link href="/auth/login" className="block">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full"
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      Login
                    </Button>
                  </Link>
                  <Link href="/auth/register" className="block">
                    <Button
                      size="sm"
                      className="w-full"
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      Sign Up
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
