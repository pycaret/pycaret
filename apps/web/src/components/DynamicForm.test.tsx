import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DynamicForm, ParamInput } from './DynamicForm';
import { applyDefaults, stripDefaults } from './DynamicForm.helpers';
import type { SetupParam, SetupParamSchema } from '@/api/types';

/**
 * These tests lock in the central contract: the dynamic form is 100%
 * data-driven. If any of these break, something has started hard-coding a
 * parameter name somewhere in the UI.
 */

const mkParam = (p: Partial<SetupParam> & Pick<SetupParam, 'name' | 'kind'>): SetupParam => ({
  default: null,
  description: '',
  choices: null,
  minimum: null,
  maximum: null,
  required: false,
  group: 'General',
  ...p,
});

// ────────────────────────────────────────────────────────── ParamInput

describe('<ParamInput>', () => {
  it('renders a switch for kind: bool and returns its boolean state', async () => {
    const onChange = vi.fn();
    render(
      <ParamInput
        param={mkParam({ name: 'normalize', kind: 'bool', default: false })}
        value={false}
        onChange={onChange}
      />,
    );
    const sw = screen.getByRole('switch');
    expect(sw).toHaveAttribute('aria-checked', 'false');
    await userEvent.click(sw);
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it('renders a number input for kind: int and parses to a number', () => {
    const onChange = vi.fn();
    render(
      <ParamInput
        param={mkParam({ name: 'fold', kind: 'int', minimum: 2, maximum: 20 })}
        value={10}
        onChange={onChange}
      />,
    );
    // Use fireEvent.change so the value is replaced atomically; userEvent.type
    // on a controlled input with a mocked onChange won't clear the DOM value.
    const input = screen.getByRole('spinbutton');
    fireEvent.change(input, { target: { value: '5' } });
    expect(onChange).toHaveBeenCalledWith(5);
    // And that the min / max attrs round-trip from the param.
    expect(input).toHaveAttribute('min', '2');
    expect(input).toHaveAttribute('max', '20');
  });

  it('clears kind: int to null when the input is emptied', () => {
    const onChange = vi.fn();
    render(
      <ParamInput
        param={mkParam({ name: 'fold', kind: 'int' })}
        value={10}
        onChange={onChange}
      />,
    );
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '' } });
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('renders a number input for kind: float with a fractional step', async () => {
    const onChange = vi.fn();
    render(
      <ParamInput
        param={mkParam({ name: 'train_size', kind: 'float', default: 0.7 })}
        value={0.7}
        onChange={onChange}
      />,
    );
    const input = screen.getByRole('spinbutton');
    expect(input).toHaveAttribute('step', '0.01');
  });

  it('renders a <select> for kind: enum with all choices', () => {
    render(
      <ParamInput
        param={mkParam({
          name: 'fold_strategy',
          kind: 'enum',
          choices: ['kfold', 'stratifiedkfold', 'groupkfold'],
          default: 'stratifiedkfold',
          required: true,
        })}
        value="stratifiedkfold"
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole('option', { name: 'kfold' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'stratifiedkfold' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'groupkfold' })).toBeInTheDocument();
    // Required enum has no "— none —" option.
    expect(screen.queryByRole('option', { name: /none/i })).not.toBeInTheDocument();
  });

  it('falls back to text input for kind: column when no columns list is supplied', () => {
    render(
      <ParamInput
        param={mkParam({ name: 'target', kind: 'column', required: true })}
        value={null}
        onChange={vi.fn()}
      />,
    );
    const input = screen.getByPlaceholderText(/column name/i);
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'text');
  });

  it('renders a <select> for kind: column when columns are supplied', () => {
    render(
      <ParamInput
        param={mkParam({ name: 'target', kind: 'column', required: true })}
        value={null}
        onChange={vi.fn()}
        columns={['age', 'income', 'churn']}
      />,
    );
    expect(screen.getByRole('option', { name: 'age' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'churn' })).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────── applyDefaults / stripDefaults

describe('applyDefaults / stripDefaults', () => {
  const schema: SetupParamSchema = {
    task: 'classification',
    groups: ['Data', 'Preprocessing'],
    parameters: [
      mkParam({ name: 'train_size', kind: 'float', default: 0.7, group: 'Data' }),
      mkParam({ name: 'fold', kind: 'int', default: 10, group: 'Data' }),
      mkParam({
        name: 'normalize',
        kind: 'bool',
        default: false,
        group: 'Preprocessing',
      }),
      mkParam({ name: 'target', kind: 'column', default: null, group: 'Data' }),
    ],
  };

  it('applies defaults but preserves user overrides', () => {
    const merged = applyDefaults(schema, { fold: 5 });
    expect(merged).toEqual({ train_size: 0.7, fold: 5, normalize: false });
  });

  it('strips values equal to defaults + empty values — only user intent survives', () => {
    const stripped = stripDefaults(schema, {
      train_size: 0.7,   // default
      fold: 5,           // overridden
      normalize: true,   // overridden
      target: '',        // empty
    });
    expect(stripped).toEqual({ fold: 5, normalize: true });
  });
});

// ─────────────────────────────────────────────────────── DynamicForm

describe('<DynamicForm>', () => {
  const schema: SetupParamSchema = {
    task: 'classification',
    groups: ['Data', 'Preprocessing'],
    parameters: [
      mkParam({ name: 'train_size', kind: 'float', default: 0.7, group: 'Data' }),
      mkParam({ name: 'fold', kind: 'int', default: 10, group: 'Data' }),
      mkParam({
        name: 'normalize',
        kind: 'bool',
        default: false,
        group: 'Preprocessing',
      }),
    ],
  };

  it('groups parameters by their `group` field, preserving schema.groups order', () => {
    render(
      <DynamicForm schema={schema} values={{}} onChange={vi.fn()} />,
    );
    const legends = screen.getAllByText(/^Data$|^Preprocessing$/).map((el) => el.textContent);
    // Data comes before Preprocessing — matches schema.groups order.
    expect(legends).toEqual(['Data', 'Preprocessing']);
  });

  it('hides parameters listed in `hide`', () => {
    render(
      <DynamicForm
        schema={schema}
        values={{}}
        onChange={vi.fn()}
        hide={['normalize']}
      />,
    );
    expect(screen.queryByText('normalize')).not.toBeInTheDocument();
    expect(screen.getByText('train_size')).toBeInTheDocument();
  });

  it('bubbles onChange with a merged values object when a single field changes', async () => {
    const onChange = vi.fn();
    render(
      <DynamicForm
        schema={schema}
        values={{ train_size: 0.7, fold: 10, normalize: false }}
        onChange={onChange}
      />,
    );
    await userEvent.click(screen.getByRole('switch'));
    expect(onChange).toHaveBeenLastCalledWith({
      train_size: 0.7,
      fold: 10,
      normalize: true,
    });
  });

  it('works with an empty schema (zero parameters)', () => {
    const empty: SetupParamSchema = {
      task: 'classification',
      groups: [],
      parameters: [],
    };
    render(<DynamicForm schema={empty} values={{}} onChange={vi.fn()} />);
    // No legends, no inputs — nothing to assert a miss against; success is "no throw".
    expect(document.body).toBeTruthy();
  });
});
