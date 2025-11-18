/**
 * SearchBar Component
 * Main search input with suggestions
 */

'use client';

import * as React from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Search, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  placeholder?: string;
  suggestions?: string[];
  className?: string;
}

export function SearchBar({
  value,
  onChange,
  onSearch,
  placeholder = 'Search for faculty, research topics, techniques...',
  suggestions = [],
  className,
}: SearchBarProps) {
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const [selectedIndex, setSelectedIndex] = React.useState(-1);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && suggestions[selectedIndex]) {
        onChange(suggestions[selectedIndex]);
      }
      onSearch();
      setShowSuggestions(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const handleClear = () => {
    onChange('');
    inputRef.current?.focus();
  };

  return (
    <div className={cn('relative w-full', className)}>
      <div className="relative flex items-center">
        <Search className="absolute left-3 w-5 h-5 text-muted-foreground pointer-events-none" />
        <Input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setShowSuggestions(true);
            setSelectedIndex(-1);
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          placeholder={placeholder}
          className="pl-10 pr-20 h-12 text-base"
        />
        {value && (
          <button
            onClick={handleClear}
            className="absolute right-16 p-1 hover:bg-accent rounded-sm"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        )}
        <Button
          onClick={onSearch}
          className="absolute right-1 h-10"
          size="sm"
        >
          Search
        </Button>
      </div>

      {/* Suggestions Dropdown */}
      {showSuggestions && suggestions.length > 0 && value && (
        <div className="absolute z-50 w-full mt-2 bg-popover border rounded-lg shadow-lg max-h-80 overflow-y-auto">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => {
                onChange(suggestion);
                onSearch();
                setShowSuggestions(false);
              }}
              className={cn(
                'w-full px-4 py-2 text-left hover:bg-accent transition-colors text-sm',
                selectedIndex === index && 'bg-accent'
              )}
            >
              <span className="flex items-center gap-2">
                <Search className="w-4 h-4 text-muted-foreground" />
                {suggestion}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
