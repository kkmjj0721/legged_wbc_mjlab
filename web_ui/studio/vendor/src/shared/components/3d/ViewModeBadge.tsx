interface ViewModeBadgeProps {
  label: string;
}

export function ViewModeBadge({ label }: ViewModeBadgeProps) {
  return (
    <div className="pointer-events-none absolute left-4 top-4 z-20 select-none">
      <div className="rounded-md border border-border-black bg-panel-bg px-2 py-1 text-[10px] font-medium leading-none text-text-secondary shadow-sm sm:text-[11px]">
        {label}
      </div>
    </div>
  );
}
