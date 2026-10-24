export const POINTER_RESIZE_START_EVENT = 'urdf-studio:pointer-resize-start';
export const POINTER_RESIZE_END_EVENT = 'urdf-studio:pointer-resize-end';

export function dispatchPointerResizeEvent(eventName: string) {
  if (typeof window === 'undefined') {
    return;
  }

  window.dispatchEvent(new Event(eventName));
}
