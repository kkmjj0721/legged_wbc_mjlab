import { translations } from './translations';
import type { Language, TranslationKeys } from './types';

const SUPPORTED_LANGUAGES: readonly Language[] = ['en', 'zh'];

export function normalizeLanguage(value: unknown): Language | null {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === 'zh' || normalized === 'zh-cn' || normalized.startsWith('zh-')) {
    return 'zh';
  }
  if (normalized === 'en' || normalized === 'en-us' || normalized.startsWith('en-')) {
    return 'en';
  }

  return SUPPORTED_LANGUAGES.includes(normalized as Language) ? (normalized as Language) : null;
}

export function resolveRuntimeLanguage(defaultLanguage: Language = 'en'): Language {
  if (typeof document !== 'undefined') {
    const documentLanguage = normalizeLanguage(
      document.documentElement.dataset.lang || document.documentElement.lang,
    );
    if (documentLanguage) {
      return documentLanguage;
    }
  }

  if (typeof window !== 'undefined') {
    const storedLanguage = normalizeLanguage(window.localStorage?.getItem('language'));
    if (storedLanguage) {
      return storedLanguage;
    }
  }

  if (typeof navigator !== 'undefined') {
    const browserLanguage = normalizeLanguage(navigator.language);
    if (browserLanguage) {
      return browserLanguage;
    }
  }

  return defaultLanguage;
}

export function getRuntimeLanguageTranslations(defaultLanguage?: Language): {
  lang: Language;
  t: TranslationKeys;
} {
  const lang = resolveRuntimeLanguage(defaultLanguage);
  return { lang, t: translations[lang] };
}
