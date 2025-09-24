import numpy as np

def preprocess_row_tess(row):
    sequence = np.array(row['Flux'])
    times = np.array(row['Time'])

    color = row['BP-RP']
    parallax = row['Parallax']
    m_G = row['M_G']
    period = row['Period']
    amp = row['Amp']

    mean_sequence = np.mean(sequence)
    std_sequence = np.std(sequence)
    mask = np.abs(sequence - mean_sequence) <= 5 * std_sequence
    filtered_sequence = sequence[mask]
    filtered_times = times[mask]

    seq_min = np.percentile(filtered_sequence, 5)
    seq_max = np.percentile(filtered_sequence, 95)
    filtered_sequence = (filtered_sequence - seq_min) / (seq_max - seq_min)

    times_min = np.min(filtered_times)
    times_max = np.max(filtered_times)
    filtered_times = (filtered_times - times_min) / (times_max - times_min)

    return {
        'input_ids': np.column_stack((filtered_times, filtered_sequence)).astype(np.float32),
        'attention_mask': np.ones(len(filtered_sequence), dtype=np.int8),
        'feature': np.array([period, amp, m_G, color, parallax], dtype=np.float32),
        'meta': {
            'Type': row['Type'],
        }
    }

def preprocess_row_ztf(row):
    sequence = np.array(row['mag'])
    times = np.array(row['hmjd'])

    color = row['BP-RP']
    parallax = row['Parallax']
    parallax_error = row['Parallax_err']
    Gmag = row['Gmag']
    period = row['Period']

    mean_sequence = np.mean(sequence)
    std_sequence = np.std(sequence)
    mask = np.abs(sequence - mean_sequence) <= 3 * std_sequence
    filtered_sequence = sequence[mask]
    filtered_times = times[mask]

    seq_min = np.percentile(filtered_sequence, 5)
    seq_max = np.percentile(filtered_sequence, 95)
    amp = seq_max - seq_min

    filtered_sequence = 10 ** (-0.4 * filtered_sequence)
    seq_min = np.percentile(filtered_sequence, 5)
    seq_max = np.percentile(filtered_sequence, 95)
    filtered_sequence = (filtered_sequence - seq_min) / (seq_max - seq_min)

    times_min = np.min(filtered_times)
    times_max = np.max(filtered_times)
    filtered_times = (filtered_times - times_min) / (times_max - times_min)

    return {
        'input_ids': np.column_stack((filtered_times, filtered_sequence)).astype(np.float32),
        'attention_mask': np.ones(len(filtered_sequence), dtype=np.int8),
        'feature': np.array([period, amp, color, parallax, parallax_error, Gmag], dtype=np.float32),
        'meta': {
            'Type': row['Type'],
        }
    }

def preprocess_row_gaia(row):
    sequence = np.array(row['g_transit_flux'])
    times = np.array(row['g_transit_time'])

    color = row['bp_rp']
    parallax = row['parallax']
    parallax_error = row['parallax_error']
    Gmag = row['phot_g_mean_mag']
    period = row['period']
    amp = row['amp']

    mean_sequence = np.mean(sequence)
    std_sequence = np.std(sequence)
    mask = np.abs(sequence - mean_sequence) <= 3 * std_sequence
    filtered_sequence = sequence[mask]
    filtered_times = times[mask]

    seq_min = np.percentile(filtered_sequence, 5)
    seq_max = np.percentile(filtered_sequence, 95)
    filtered_sequence = (filtered_sequence - seq_min) / (seq_max - seq_min)

    times_min = np.min(filtered_times)
    times_max = np.max(filtered_times)
    filtered_times = (filtered_times - times_min) / (times_max - times_min)

    return {
        'input_ids': np.column_stack((filtered_times, filtered_sequence)).astype(np.float32),
        'attention_mask': np.ones(len(filtered_sequence), dtype=np.int8),
        'feature': np.array([parallax, parallax_error, color, period, amp, Gmag], dtype=np.float32),
        'meta': {
            'Type': row['best_class_name'],
        }
    }