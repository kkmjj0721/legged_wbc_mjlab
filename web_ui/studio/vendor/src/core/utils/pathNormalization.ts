/**
 * Normalize asset-style relative paths for parser and export code.
 *
 * Leading roots are not preserved: "/a/b" normalizes to "a/b". Parent
 * traversal only collapses an existing segment, so leading ".." segments are
 * ignored instead of being retained.
 */
export function normalizeRelativePath(path: string): string {
  const segments = path.replace(/\\/g, '/').split('/');
  const stack: string[] = [];

  for (const segment of segments) {
    if (!segment || segment === '.') {
      continue;
    }

    if (segment === '..') {
      if (stack.length > 0) {
        stack.pop();
      }
      continue;
    }

    stack.push(segment);
  }

  return stack.join('/');
}
