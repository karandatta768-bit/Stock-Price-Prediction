let counter = 0;

export function createId(prefix) {
  counter += 1;
  return `${prefix}-${Date.now()}-${counter}`;
}
