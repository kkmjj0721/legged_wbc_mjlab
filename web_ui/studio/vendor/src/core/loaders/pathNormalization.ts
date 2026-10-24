// PERFORMANCE: Optimized cleanFilePath without array allocations
// Uses string manipulation instead of split/reduce/join
export const cleanFilePath = (path: string): string => {
    // Fast path: no special characters
    if (!path.includes('..') && !path.includes('./') && !path.includes('\\')) {
        return path.replace(/\/+/g, '/');
    }

    // Normalize backslashes first
    let result = path.replace(/\\/g, '/');

    // Remove ./ references
    result = result.replace(/\/\.\//g, '/').replace(/^\.\//g, '');

    // Handle .. by iterative replacement (avoids array allocation)
    let prev = '';
    while (prev !== result) {
        prev = result;
        result = result.replace(/\/[^\/]+\/\.\.\//, '/');
        result = result.replace(/\/[^\/]+\/\.\.$/g, '');
    }

    // Clean up any leading ../
    result = result.replace(/^\.\.\/+/g, '');

    // Normalize multiple slashes
    result = result.replace(/\/+/g, '/');

    return result;
};
