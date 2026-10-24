/**
 * i18n Module
 * Internationalization support for the application
 */

export type { Language, TranslationKeys, Translations } from './types';
export { translations } from './translations';
export { en, zh } from './locales';
export {
  getRuntimeLanguageTranslations,
  normalizeLanguage,
  resolveRuntimeLanguage,
} from './runtimeLanguage';
