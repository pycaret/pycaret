import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Leaderboard } from './Leaderboard';

/**
 * The leaderboard must render *any* JSON-table shape the engine emits —
 * no hard-coded column names. These tests lock that down.
 */

describe('<Leaderboard>', () => {
  it('renders the empty-state hint for null / empty rows', () => {
    const { rerender } = render(<Leaderboard rows={null} />);
    expect(screen.getByText(/leaderboard will appear here/i)).toBeInTheDocument();
    rerender(<Leaderboard rows={[]} />);
    expect(screen.getByText(/leaderboard will appear here/i)).toBeInTheDocument();
  });

  it('renders every column from the first row, preserving engine order', () => {
    render(
      <Leaderboard
        rows={[
          { index: 0, Model: 'Logistic Regression', Accuracy: 0.92, AUC: 0.95 },
          { index: 1, Model: 'Random Forest', Accuracy: 0.9, AUC: 0.94 },
        ]}
      />,
    );
    const headerCells = screen.getAllByRole('columnheader').map((c) => c.textContent);
    // Each cell's textContent might include the sort-indicator; strip it.
    expect(headerCells.map((t) => t?.replace(/[▼▲]/g, '').trim())).toEqual([
      'index',
      'Model',
      'Accuracy',
      'AUC',
    ]);
  });

  it('formats numbers: integers stay bare, floats show 4 decimals', () => {
    render(
      <Leaderboard rows={[{ index: 0, Model: 'LR', Accuracy: 0.923456, rank: 1 }]} />,
    );
    // Float rendered with 4 decimals
    expect(screen.getByText('0.9235')).toBeInTheDocument();
    // Integer stays bare
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  it('sorts numerically descending on first click, ascending on second', async () => {
    const user = userEvent.setup();
    render(
      <Leaderboard
        rows={[
          { Model: 'A', Accuracy: 0.5 },
          { Model: 'B', Accuracy: 0.9 },
          { Model: 'C', Accuracy: 0.1 },
        ]}
      />,
    );

    // Sort by Accuracy — click on the header cell's visible text
    const accHeader = screen.getByRole('columnheader', { name: /Accuracy/ });
    await user.click(accHeader);
    // First data-row cell in Accuracy column should be 0.9 (desc)
    const firstRowCells = screen
      .getAllByRole('row')
      .slice(1)  // skip thead
      .map((r) => r.querySelectorAll('td')[1]?.textContent);
    expect(firstRowCells[0]).toBe('0.9000');

    await user.click(accHeader);
    const firstRowAsc = screen
      .getAllByRole('row')
      .slice(1)
      .map((r) => r.querySelectorAll('td')[1]?.textContent);
    expect(firstRowAsc[0]).toBe('0.1000');
  });
});
