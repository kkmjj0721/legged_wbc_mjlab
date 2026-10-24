export function attachContextMenuBlocker(target: EventTarget | null | undefined): () => void {
  if (!target) {
    return () => {};
  }

  const handleContextMenu = (event: Event) => {
    event.preventDefault();
  };

  const handleDragStart = (event: Event) => {
    event.preventDefault();
  };

  const handleSelectStart = (event: Event) => {
    event.preventDefault();
  };

  const handleGestureStart = (event: Event) => {
    event.preventDefault();
  };

  target.addEventListener('contextmenu', handleContextMenu);
  target.addEventListener('dragstart', handleDragStart);
  target.addEventListener('selectstart', handleSelectStart);
  target.addEventListener('gesturestart', handleGestureStart as EventListener);

  return () => {
    target.removeEventListener('contextmenu', handleContextMenu);
    target.removeEventListener('dragstart', handleDragStart);
    target.removeEventListener('selectstart', handleSelectStart);
    target.removeEventListener('gesturestart', handleGestureStart as EventListener);
  }
}
