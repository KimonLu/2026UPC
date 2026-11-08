"""Regenerate or verify the numerical tables embedded in the Section 6 TeX."""

from scripts.part06.part6_common import CONFIG, DATA, read_json


def table(caption, label, alignment, headers, rows, *, compact=False):
    return '\n'.join([
        r'\begin{table}[!htbp]', r'\centering', r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\scriptsize' if compact else r'\small',
        r'\setlength{\tabcolsep}{2.25pt}' if compact else r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{@{}' + alignment + '@{}}',
        r'\toprule', ' & '.join(headers) + r' \\', r'\midrule',
        *(' & '.join(row) + r' \\' for row in rows),
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ])


def v2_table():
    data = read_json(DATA / 'v2_summary.json')
    rows = []
    for name in CONFIG['synthetic']['models']:
        result = data[f'reference/{name}/0.005']
        position = result['position_m']
        time = result['arrival_s']
        rows.append([name,
                     f'{result["both_cross"]}/{result["n_eligible"]}',
                     *(f'{position[key]:.3f}' for key in ('median', 'p90', 'rmse')),
                     *(f'{time[key]*1000:.1f}' for key in ('median', 'p90', 'bias')),
                     f'{result["event_accuracy"]*100:.1f}'])
    return table(
        r'Synthetic predictions at \(\sigma=5\,\mathrm{mm}\). The spatial '
        r'statistics and time errors use paired crossings. Time medians '
        r'and percentiles are absolute. Bias is prediction minus truth. '
        r'Event accuracy uses all eligible trials.',
        'tab:part6-v2', 'lrrrrrrrr',
        ['Model', 'Paired', r'\multicolumn{3}{c}{Position error (m)}',
         r'\multicolumn{3}{c}{Time error (ms)}', r'Accuracy (\%)'],
        rows,
    ).replace(r'\midrule', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}'
              '\n & & Median & P90 & RMSE & Median & P90 & Bias & ' + r'\\'
              '\n' + r'\midrule', 1)


def v3_table():
    data = read_json(DATA / 'v3_summary.json')
    ref = read_json(DATA / 'v2_summary.json')['reference/M-DM/0']
    descriptors = [
        ('Reference truth', 'M-DM', ref),
        ('Crosswind', 'M-DM', data['sidewind/M-DM/0']),
        (r'True \(C_D=0.30\)', 'M-DM', data['higher_drag/M-DM/0']),
        (r'True \(C_D=0.30\)', r'M-DM, \(C_D=0.20\)', data['higher_drag/M-DM fixed drag/0']),
        ('Spin decay', 'M-DM', data['spin_decay/M-DM/0']),
        (r'Axis rotation about \(\mathbf e_z\)', 'M-DM', data['axis_z_control/M-DM/0']),
        (r'Axis rotation about \(\mathbf e_y\)', 'M-DM', data['axis_y/M-DM/0']),
        ('Zero spin with wake', 'M-D', data['wake/M-D/0']),
        ('Zero spin with wake', 'M-D + known wake', data['wake/M-D + known wake/0']),
    ]
    rows = []
    for scenario, model, result in descriptors:
        position, time = result['position_m'], result['arrival_s']
        spatial = [f'{position[key]*1000:.2f}' if position[key] >= .000005 else r'\(<0.01\)'
                   for key in ('median', 'p90')]
        temporal = f'{time["median"]*1000:.3f}' if time['median'] >= .0000005 else r'\(<0.001\)'
        rows.append([scenario, model, str(result['both_cross']), *spatial, temporal,
                     f'{result["n_eligible"]-result["correct_events"]}/{result["n_eligible"]}'])
    return table(
        'Noiseless model-mismatch tests. Position and time errors use '
        'paired crossings. Time error is absolute. Event disagreements '
        'are counted over all eligible trajectories.',
        'tab:part6-v3', 'llrrrrr',
        ['Truth', 'Fit', 'Paired', r'\multicolumn{2}{c}{Position (mm)}',
         r'\shortstack{Time\\(ms)}', 'Event errors'],
        rows, compact=True,
    ).replace(r'\midrule', r'\cmidrule(lr){4-5}' +
              '\n & & & Median & P90 & Median & ' + r'\\' + '\n' + r'\midrule', 1)


def shot_table():
    cases = read_json(DATA / 'given_shot.json')['cases']
    rows = []
    for case_id, case in cases.items():
        label = '4A' if case_id.startswith('4A/') else case_id
        if case_id == '4A/x+2':
            label = r'\textbf{4A}'
        if case_id in ('M-DM5', 'M-DM1'):
            label = 'M-DM'
        if case_id == 'M-D20':
            label = 'M-D (fixed)'
        v0 = ','.join(f'{v:.2f}' for v in case['v0_m_s'])
        spin = ','.join(f'{v:.1f}' for v in case['omega_basis_rad_s'][1:])
        cd = '--' if case['cd'] is None else f'{case["cd"]:.3f}'
        tau = f'{case["tau_s"]:g}' if case['tau_s'] else r'\(\infty\)' if case['k_l'] == 1 else '--'
        rms = case['rms_m']*1000
        rms_text = r'\(<0.01\)' if case['k_l'] == 1 and rms < .01 else f'{rms:.2f}'
        point = case['goal']
        wx, wy, _ = case['wind_m_s']
        rows.append([label, rf'\(({wx:g},{wy:g})\)', tau, cd, rf'\(({v0})\)',
                     rf'\(({spin})\)' if case['k_l'] == 1 else '--', rms_text,
                     f'{point["t_s"]:.4f}', rf'\(({point["y_m"]:.3f},{point["z_m"]:.3f})\)'])
    return table(
        r'Given-shot fits. All rows cross inside the goal before ground contact. '
        r'RMS is computed over the six later coordinates. The fixed M-D row prescribes \(C_D=0.20\). '
        r'Wind has \(w_z=0\), '
        r'and the calm M-DM row also represents 4A in calm air. '
        r'Spin components use the chord basis with \(\omega_1=0\) and \(k_L=1\). '
        r'B-CA also fits \(\mathbf a=(-16,-16,-12)\,\mathrm{m\,s^{-2}}\), '
        r'and B-G fixes \(\mathbf a=(0,0,-g)\).',
        'tab:part6-shot', 'lccrccrrc',
        ['Model', r'\shortstack{\((w_x,w_y)\)\\(\(\mathrm{m/s}\))}',
         r'\shortstack{\(\tau\)\\(s)}', r'\(C_D\)',
         r'\shortstack{\(\mathbf v_0\)\\(\(\mathrm{m/s}\))}',
         r'\shortstack{\((\omega_2,\omega_3)\)\\(\(\mathrm{rad/s}\))}',
         r'\shortstack{RMS\\(mm)}', r'\shortstack{\(t_g\)\\(s)}',
         r'\shortstack{\((y_g,z_g)\)\\(m)}'], rows, compact=True,
    )


def main():
    tables = {"v2": v2_table(), "v3": v3_table(), "shot": shot_table()}
    for name, table in tables.items():
        (DATA / f"{name}_table.tex").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
