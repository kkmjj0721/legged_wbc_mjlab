import React from 'react';

import { Switch, type SwitchProps } from './Switch';

export type CompactSwitchProps = Omit<SwitchProps, 'size'>;

export const CompactSwitch: React.FC<CompactSwitchProps> = (props) => (
  <Switch size="sm" {...props} />
);
