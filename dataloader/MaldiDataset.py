import numpy as np
import pandas as pd
import os
import pandas as pd
import numpy as np
from dataloader.preprocess import SequentialPreprocessor
from dataloader.SpectrumObject import SpectrumObject
from tqdm import tqdm
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# Base from maldi-nn, thanks to the authors. Adapted and expanded by Alejandro Guerrero-López.

   
class MaldiDataset(Dataset):
    def __init__(self, root_dir, preprocess_pipeline: SequentialPreprocessor = None, taxa: list = None, n_samples: int = None):
        self.root_dir = root_dir
        self.preprocess_pipeline = preprocess_pipeline
        self.data = []
        self.taxa = taxa  # List of genus or genus-species ("Genus species")
        self.n_samples = n_samples

    def parse_dataset(self):
        print(f"Reading dataset from {self.root_dir}")

        def should_include(genus, species):
            if not self.taxa:
                return True

            full_name = f"{genus} {species}"

            for taxon in self.taxa:
                if " " in taxon:  # Species level filtering
                    if full_name.lower() == taxon.lower():
                        return True
                else:  # Genus level filtering
                    if genus.lower() == taxon.lower():
                        return True
            return False

        total_folders = sum(
            1 for year in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, year))
            for genus in os.listdir(os.path.join(self.root_dir, year)) if os.path.isdir(os.path.join(self.root_dir, year, genus))
            for species in os.listdir(os.path.join(self.root_dir, year, genus)) if os.path.isdir(os.path.join(self.root_dir, year, genus, species))
            for replicate in os.listdir(os.path.join(self.root_dir, year, genus, species)) if os.path.isdir(os.path.join(self.root_dir, year, genus, species, replicate))
        )

        with tqdm(total=total_folders, desc="Processing Dataset", unit="folder") as pbar:
            for year in os.listdir(self.root_dir):
                year_path = os.path.join(self.root_dir, year)
                if os.path.isdir(year_path):
                    for genus in os.listdir(year_path):
                        genus_path = os.path.join(year_path, genus)
                        if os.path.isdir(genus_path):
                            for species in os.listdir(genus_path):
                                species_path = os.path.join(genus_path, species)
                                if os.path.isdir(species_path) and should_include(genus, species):
                                    genus_species_label = f"{genus} {species}"
                                    for replicate in os.listdir(species_path):
                                        id = replicate
                                        replicate_path = os.path.join(species_path, replicate)
                                        if os.path.isdir(replicate_path):
                                            for lecture in os.listdir(replicate_path):
                                                lecture_path = os.path.join(replicate_path, lecture)
                                                if os.path.isdir(lecture_path):
                                                    acqu_file, fid_file = self._find_acqu_fid_files(lecture_path)
                                                    if acqu_file and fid_file:
                                                        spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
                                                        if self.preprocess_pipeline:
                                                            spectrum = self.preprocess_pipeline(spectrum)
                                                        if np.isnan(spectrum.intensity).any():
                                                            continue
                                                        self.data.append({
                                                            'id': id,
                                                            'spectrum_intensity': spectrum.intensity,
                                                            'spectrum_mz': spectrum.mz,
                                                            'year_label': year,
                                                            'genus_label': genus,
                                                            'genus_species_label': genus_species_label,
                                                        })
                                            pbar.update(1)

        if self.n_samples and len(self.data) > self.n_samples:
            print(f"Subsampling to {self.n_samples} samples from {len(self.data)}.")
            np.random.shuffle(self.data)
            self.data = self.data[:self.n_samples]

    def __len__(self):
        return len(self.data)
    
    def _find_acqu_fid_files(self, directory):
        acqu_file = None
        fid_file = None
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'acqu':
                    acqu_file = os.path.join(root, file)
                elif file == 'fid':
                    fid_file = os.path.join(root, file)
                if acqu_file and fid_file:
                    return acqu_file, fid_file
        return acqu_file, fid_file

    def get_data(self):
        return self.data
    
    def pad_collate_fn(batch):
        """
        Pads the 'intensity' and 'mz' tensors to the length of the longest one in the batch.
        Other fields are collected into lists.
        """
        # Determine the maximum length in the current batch.
        max_len = max(item['intensity'].size(0) for item in batch)
        
        intensities = []
        mzs = []
        year_labels = []
        genus_labels = []
        species_labels = []
        
        for item in batch:
            intensity = item['intensity']
            mz = item['mz']
            pad_size = max_len - intensity.size(0)
            # Pad at the end with zeros.
            intensity_padded = F.pad(intensity, (0, pad_size), mode='constant', value=0)
            mz_padded = F.pad(mz, (0, pad_size), mode='constant', value=0)
            intensities.append(intensity_padded)
            mzs.append(mz_padded)
            year_labels.append(item['year_label'])
            genus_labels.append(item['genus_label'])
            species_labels.append(item['genus_species_label'])
            
        batch_intensity = torch.stack(intensities)  # [batch_size, max_len]
        batch_mz = torch.stack(mzs)               # [batch_size, max_len]
        
        return {
            'intensity': batch_intensity,
            'mz': batch_mz,
            'year_label': year_labels,
            'genus_label': genus_labels,
            'genus_species_label': species_labels
        }
    
    def __getitem__(self, idx):
        # Get the dictionary for the sample.
        item = self.data[idx]
        
        # Convert the numpy arrays to torch tensors.
        intensity = torch.tensor(item['spectrum_intensity'], dtype=torch.float32)
        mz = torch.tensor(item['spectrum_mz'], dtype=torch.float32)

        return intensity, mz, item['year_label'], item['genus_label'], item['genus_species_label']

    def _parse_folder_name(self, folder_name):
        # Split folder name into genus, species, and hospital code
        parts = folder_name.split()
        genus_species = " ".join(parts[:2])
        hospital_code = " ".join(parts[2:])
        return genus_species, hospital_code
    

    def save_to_hdf5(self, file_name):
        with h5py.File(file_name, 'w') as h5f:
            spectra = np.array([d['spectrum_intensity'] for d in self.data]).astype(np.float64)
            mz_values = np.array([d['spectrum_mz'] for d in self.data])
            year_labels = np.array([d['year_label'] for d in self.data])
            genus_labels = np.array([d['genus_label'] for d in self.data])
            genus_species_labels = np.array([d['genus_species_label'] for d in self.data])

            h5f.create_dataset('spectra', data=spectra, chunks=True, compression='gzip')
            h5f.create_dataset('mz_values', data=mz_values, chunks=True, compression='gzip')
            h5f.create_dataset('year_labels', data=year_labels.astype('S'), chunks=True, compression='gzip')
            h5f.create_dataset('genus_labels', data=genus_labels.astype('S'), chunks=True, compression='gzip')
            h5f.create_dataset('genus_species_labels', data=genus_species_labels.astype('S'), chunks=True, compression='gzip')

    def load_from_hdf5(self, file_name):
        with h5py.File(file_name, 'r') as h5f:
            self.data = [{
                'spectrum_intensity': h5f['spectra'][i],
                'spectrum_mz': h5f['mz_values'][i],
                'year_label': h5f['year_labels'][i].decode('utf-8'),
                'genus_label': h5f['genus_labels'][i].decode('utf-8'),
                'genus_species_label': h5f['genus_species_labels'][i].decode('utf-8'),
            } for i in range(len(h5f['spectra']))]

    def get_data(self):
        return self.data
    