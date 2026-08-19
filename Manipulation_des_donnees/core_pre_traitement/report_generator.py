import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


class ReportGenerator:
    """
    Produit un rapport JSON et un rapport texte à partir des résultats
    de preprocessing, avec des sections spécifiques à chaque dataset.
    """

    REFERENCE = (
        "Vesselness Filters: A Survey with Benchmarks "
        "Applied to Liver Imaging (ICPR 2020)"
    )

    def __init__(self, output_dir: Path, target_spacing: tuple):
        self.output_dir = Path(output_dir)
        self.target_spacing = target_spacing
        self._stats_cache: Dict[str, Dict] = {}



    def generate(self, results: list[dict]) -> None:
        """Génère et sauvegarde les rapports."""
        # Grouper par dataset
        grouped = self._group_by_dataset(results)
        
        # Rapport global
        self._generate_global_report(results, grouped)
        
        # Rapport par dataset
        for dataset_name, dataset_results in grouped.items():
            self._generate_dataset_report(dataset_name, dataset_results)
        
        # Résumé console
        self._print_summary(results, grouped)



    def _group_by_dataset(self, results: list[dict]) -> Dict[str, list[dict]]:
        """Regroupe les résultats par dataset."""
        grouped: Dict[str, list[dict]] = {}
        for r in results:
            dataset = r.get("dataset", "unknown")
            grouped.setdefault(dataset, []).append(r)
        return grouped


    def _generate_global_report(self, results: list[dict], grouped: Dict[str, list[dict]]) -> None:
        """Rapport global consolidé."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        total_time = sum(r.get("time_seconds", 0) for r in successful)

        dataset_stats = {}
        for name, group in grouped.items():
            ok = [r for r in group if r["status"] == "success"]
            dataset_stats[name] = {
                "total": len(group),
                "successful": len(ok),
                "failed": len(group) - len(ok),
                "time_total": sum(r.get("time_seconds", 0) for r in ok),
                "time_avg": np.mean([r.get("time_seconds", 0) for r in ok]) if ok else 0,
            }

        report = {
            "title": "Multi-Dataset Preprocessing Report",
            "date": datetime.now().isoformat(),
            "reference": self.REFERENCE,
            "target_spacing": self.target_spacing,
            "summary": {
                "total_patients": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "total_time_seconds": total_time,
                "avg_time_per_patient": total_time / max(len(successful), 1),
            },
            "datasets": dataset_stats,
            "results": successful,
            "errors": failed,
        }

        self._save_json(report, "global_preprocessing_report.json")
        self._save_global_txt(report, successful, failed, grouped)

    def _save_global_txt(self, report: dict, successful: list, failed: list, grouped: dict) -> None:
        """Sauvegarde le rapport global en TXT."""
        path = self.output_dir / "global_preprocessing_report.txt"
        sep = "="* 80

        with open(path, "w") as f:
            f.write(f"{sep}\n{report['title']}\n{sep}\n\n")
            f.write(f"Date : {report['date']}\n")
            f.write(f"Reference : {report['reference']}\n")
            f.write(f"Spacing : {report['target_spacing']}\n\n")

            s = report["summary"]
            f.write("-"* 40 + "\nGLOBAL SUMMARY\n"+ "-"* 40 + "\n")
            f.write(f"Total patients : {s['total_patients']}\n")
            f.write(f"Successfully processed: {s['successful']}\n")
            f.write(f"Failed : {s['failed']}\n")
            f.write(f"Total time : {s['total_time_seconds']:.2f} s\n")
            f.write(f"Average per patient : {s['avg_time_per_patient']:.2f} s\n\n")

            f.write("-"* 40 + "\nSUMMARY BY DATASET\n"+ "-"* 40 + "\n")
            for name, stats in report["datasets"].items():
                f.write(f"\n{name.upper()}\n")
                f.write(f"Patients : {stats['total']}\n")
                f.write(f"Successful : {stats['successful']}\n")
                f.write(f"Failed : {stats['failed']}\n")
                f.write(f"Total time : {stats['time_total']:.2f} s\n")
                f.write(f"Avg time : {stats['time_avg']:.2f} s\n")

            f.write(f"\n{sep}\nDETAILED RESULTS BY DATASET\n{sep}\n")
            for dataset_name, dataset_results in grouped.items():
                f.write(f"\n{'='* 60}\n{dataset_name.upper()}\n{'='* 60}\n")
                self._write_dataset_results(f, dataset_name, dataset_results)

            # Errors
            if failed:
                f.write(f"\n{sep}\nERRORS\n{sep}\n")
                for e in failed:
                    f.write(f"{e.get('patient_id', 'unknown')}: {e.get('error', 'Unknown')}\n")

    def _generate_dataset_report(self, dataset_name: str, results: list[dict]) -> None:
        """Génère un rapport détaillé pour un dataset spécifique."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        stats = self._compute_dataset_stats(dataset_name, successful)
        
        report = {
            "title": f"{dataset_name.upper()} Preprocessing Report",
            "date": datetime.now().isoformat(),
            "dataset": dataset_name,
            "reference": self.REFERENCE,
            "target_spacing": self.target_spacing,
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "total_time": sum(r.get("time_seconds", 0) for r in successful),
                "avg_time": np.mean([r.get("time_seconds", 0) for r in successful]) if successful else 0,
            },
            "statistics": stats,
            "results": successful,
            "errors": failed,
        }

        self._save_json(report, f"{dataset_name}_preprocessing_report.json")
        self._save_dataset_txt(dataset_name, report)

    def _compute_dataset_stats(self, dataset_name: str, results: list[dict]) -> Dict:
        """Calcule des statistiques spécifiques au dataset."""
        stats: Dict[str, Any] = {"count": len(results)}

        if not results:
            return stats

        # Statistiques communes
        spacings = [r.get("original_spacing") for r in results if r.get("original_spacing")]
        if spacings:
            avg_spacing = np.mean(spacings, axis=0)
            stats["avg_original_spacing"] = list(avg_spacing)
            stats["spacing_std"] = list(np.std(spacings, axis=0))

        sizes = [r.get("original_size") for r in results if r.get("original_size")]
        if sizes:
            avg_size = np.mean(sizes, axis=0)
            stats["avg_original_size"] = [int(x) for x in avg_size]

        # Statistiques spécifiques par dataset
        if dataset_name == "ircad":
            stats.update(self._stats_ircad(results))
        elif dataset_name == "bullitt":
            stats.update(self._stats_bullitt(results))
        elif dataset_name == "sennet":
            stats.update(self._stats_sennet(results))
        elif dataset_name == "vascusynth":
            stats.update(self._stats_vascusynth(results))

        return stats



    def _stats_ircad(self, results: list[dict]) -> Dict:
        """Statistiques spécifiques IRCAD."""
        stats = {
            "patients_count": len(results),
            "liver_mask_applied": results[0].get("version") == "in_liver"if results else False,
        }
        
        # Vérifier la présence des fichiers
        has_vessels = sum(1 for r in results if r.get("processed_files", {}).get("vessels_gt"))
        stats["has_vessels_gt"] = has_vessels
        stats["vessels_gt_rate"] = has_vessels / max(len(results), 1)
        
        return stats

    def _stats_bullitt(self, results: list[dict]) -> Dict:
        """Statistiques spécifiques Bullitt."""
        stats = {
            "patients_count": len(results),
            "normalization_applied": results[0].get("normalization_applied", False) if results else False,
            "cropping_applied": results[0].get("cropping_applied", False) if results else False,
            "masking_applied": results[0].get("masking_applied", False) if results else False,
        }
        
        # Vérifier la présence des fichiers
        has_labels = sum(1 for r in results if r.get("processed_files", {}).get("label"))
        has_masks = sum(1 for r in results if r.get("processed_files", {}).get("mask"))
        stats["has_vessel_labels"] = has_labels
        stats["has_brain_masks"] = has_masks
        stats["label_rate"] = has_labels / max(len(results), 1)
        stats["mask_rate"] = has_masks / max(len(results), 1)
        
        return stats

    def _stats_sennet(self, results: list[dict]) -> Dict:
        """Statistiques spécifiques SenNet."""
        stats = {
            "patients_count": len(results),
            "dataset_name": results[0].get("dataset", "sennet") if results else "sennet",
        }
        
        # Mode crop vs volume entier
        crop_applied = results[0].get("crop_applied", False) if results else False
        stats["crop_applied"] = crop_applied
        
        if crop_applied:
            stats["crop_size"] = results[0].get("crop_size", "N/A") if results else "N/A"
            stats["crop_normalize"] = results[0].get("normalization_applied", False) if results else False
        else:
            stats["normalization_applied"] = results[0].get("normalization_applied", False) if results else False
            stats["normalization_method"] = results[0].get("normalization_method", "none") if results else "none"
            stats["normalization_clip"] = results[0].get("normalization_clip_bounds", [0, 65535]) if results else [0, 65535]
        
        return stats

    def _stats_vascusynth(self, results: list[dict]) -> Dict:
        """Statistiques spécifiques VascuSynth."""
        stats = {
            "patients_count": len(results),
            "noise_levels": results[0].get("noise_levels", [5.0, 10.0, 20.0]) if results else [],
            "gt_mask_generated": results[0].get("gt_mask_generated", False) if results else False,
        }
        
        # Statistiques sur les bifurcations
        bif_counts = [r.get("bifurcations", 0) for r in results]
        if bif_counts:
            stats["bifurcations_min"] = min(bif_counts)
            stats["bifurcations_max"] = max(bif_counts)
            stats["bifurcations_mean"] = np.mean(bif_counts)
            stats["bifurcations_std"] = np.std(bif_counts)
            stats["bifurcation_complexities"] = sorted(set(bif_counts))
        
        # ROIs générées
        stats["bifurcation_mask_generated"] = results[0].get("bifurcation_mask_generated", False) if results else False
        stats["neighborhood_mask_generated"] = results[0].get("neighborhood_mask_generated", False) if results else False
        
        # Groupes et data
        groups = sorted(set(r.get("group", 0) for r in results))
        data_indices = sorted(set(r.get("data", 0) for r in results))
        stats["groups"] = groups
        stats["data_indices"] = data_indices
        
        return stats


    def _save_dataset_txt(self, dataset_name: str, report: dict) -> None:
        """Sauvegarde le rapport TXT pour un dataset spécifique."""
        path = self.output_dir / f"{dataset_name}_preprocessing_report.txt"
        sep = "="* 80

        with open(path, "w") as f:
            f.write(f"{sep}\n{report['title']}\n{sep}\n\n")
            f.write(f"Date : {report['date']}\n")
            f.write(f"Reference : {report['reference']}\n")
            f.write(f"Spacing : {report['target_spacing']}\n\n")

            # Summary
            s = report["summary"]
            f.write("-"* 40 + "\nSUMMARY\n"+ "-"* 40 + "\n")
            f.write(f"Total patients : {s['total']}\n")
            f.write(f"Successfully processed: {s['successful']}\n")
            f.write(f"Failed : {s['failed']}\n")
            f.write(f"Total time : {s['total_time']:.2f} s\n")
            f.write(f"Average per patient : {s['avg_time']:.2f} s\n\n")

            # Dataset-specific statistics
            stats = report.get("statistics", {})
            f.write("-"* 40 + f"\n{dataset_name.upper()} STATISTICS\n"+ "-"* 40 + "\n")
            self._write_dataset_stats(f, dataset_name, stats)

            # Detailed patient results
            f.write(f"\n{sep}\nPATIENT DETAILS\n{sep}\n")
            self._write_dataset_results(f, dataset_name, report["results"])

            # Errors
            if report["errors"]:
                f.write(f"\n{sep}\nERRORS\n{sep}\n")
                for e in report["errors"]:
                    f.write(f"{e.get('patient_id', 'unknown')}: {e.get('error', 'Unknown')}\n")

    def _write_dataset_stats(self, f, dataset_name: str, stats: dict) -> None:
        """Écrit les statistiques spécifiques au dataset."""
        if dataset_name == "ircad":
            f.write(f"Patients count : {stats.get('patients_count', 'N/A')}\n")
            f.write(f"Liver mask applied : {'OUI'if stats.get('liver_mask_applied') else 'NON'}\n")
            f.write(f"Has vessels GT : {stats.get('has_vessels_gt', 0)}/{stats.get('patients_count', 1)}\n")
            f.write(f"Vessels GT rate : {stats.get('vessels_gt_rate', 0):.1%}\n")

        elif dataset_name == "bullitt":
            f.write(f"Patients count : {stats.get('patients_count', 'N/A')}\n")
            f.write(f"Normalization applied: {'OUI'if stats.get('normalization_applied') else 'NON'}\n")
            f.write(f"Cropping applied : {'OUI'if stats.get('cropping_applied') else 'NON'}\n")
            f.write(f"Masking applied : {'OUI'if stats.get('masking_applied') else 'NON'}\n")
            f.write(f"Has vessel labels : {stats.get('has_vessel_labels', 0)}/{stats.get('patients_count', 1)}\n")
            f.write(f"Has brain masks : {stats.get('has_brain_masks', 0)}/{stats.get('patients_count', 1)}\n")
            f.write(f"Label rate : {stats.get('label_rate', 0):.1%}\n")
            f.write(f"Mask rate : {stats.get('mask_rate', 0):.1%}\n")

        elif dataset_name == "sennet":
            f.write(f"Dataset : {stats.get('dataset_name', 'N/A')}\n")
            f.write(f"Mode : {'CROP'if stats.get('crop_applied') else 'VOLUME ENTIER'}\n")
            if stats.get('crop_applied'):
                f.write(f"Crop size : {stats.get('crop_size', 'N/A')}³\n")
                f.write(f"Crop normalization : {'OUI'if stats.get('crop_normalize') else 'NON'}\n")
            else:
                f.write(f"Normalization : {'OUI'if stats.get('normalization_applied') else 'NON'}\n")
                f.write(f"Normalization method : {stats.get('normalization_method', 'none')}\n")
                f.write(f"Clip bounds : {stats.get('normalization_clip', [0, 65535])}\n")

        elif dataset_name == "vascusynth":
            f.write(f"Patients count : {stats.get('patients_count', 'N/A')}\n")
            f.write(f"Noise levels : {stats.get('noise_levels', [])}\n")
            f.write(f"GT mask generated : {'OUI'if stats.get('gt_mask_generated') else 'NON'}\n")
            f.write(f"Bifurcation mask : {'OUI'if stats.get('bifurcation_mask_generated') else 'NON'}\n")
            f.write(f"Neighborhood mask : {'OUI'if stats.get('neighborhood_mask_generated') else 'NON'}\n")
            if stats.get('bifurcation_complexities'):
                f.write(f"Bifurcation complexities: {stats['bifurcation_complexities']}\n")
                f.write(f"Min: {stats.get('bifurcations_min', 0)}\n")
                f.write(f"Max: {stats.get('bifurcations_max', 0)}\n")
                f.write(f"Mean: {stats.get('bifurcations_mean', 0):.1f}\n")
                f.write(f"Std: {stats.get('bifurcations_std', 0):.1f}\n")
            if stats.get('groups'):
                f.write(f"Groups : {stats['groups']}\n")
            if stats.get('data_indices'):
                f.write(f"Data indices : {stats['data_indices']}\n")

        # Statistiques communes
        if stats.get("avg_original_spacing"):
            f.write(f"\nAvg original spacing: ({stats['avg_original_spacing'][0]:.3f}, "
                    f"{stats['avg_original_spacing'][1]:.3f}, {stats['avg_original_spacing'][2]:.3f}) mm\n")
        if stats.get("avg_original_size"):
            f.write(f"Avg original size : {stats['avg_original_size']}\n")

    def _write_dataset_results(self, f, dataset_name: str, results: list[dict]) -> None:
        """Écrit les résultats détaillés des patients."""
        for r in results:
            f.write(f"\nPatient : {r['patient_id']}\n")
            f.write(f"Time : {r['time_seconds']:.2f} s\n")
            
            # Mode
            if r.get("crop_applied", False):
                f.write(f"Mode : CROP\n")
                f.write(f"Crop size : {r.get('crop_size', 'N/A')}³\n")
                f.write(f"Crop offset : {r.get('crop_offset', 'N/A')}\n")
                f.write(f"Normalization : {'OUI'if r.get('crop_normalize', False) else 'NON'}\n")
            else:
                f.write(f"Mode : VOLUME ENTIER\n")
            
            # Dataset-specific info
            if dataset_name == "ircad":
                f.write(f"Version : {r.get('version', 'original')}\n")
            elif dataset_name == "bullitt":
                f.write(f"Normalization : {'OUI'if r.get('normalization_applied', False) else 'NON'}\n")
            elif dataset_name == "vascusynth":
                f.write(f"Group/Data : {r.get('group', 'N/A')}/{r.get('data', 'N/A')}\n")
                f.write(f"Bifurcations : {r.get('bifurcations', 'N/A')}\n")
                if r.get("noise_levels"):
                    f.write(f"Noise levels : {r['noise_levels']}\n")
            
            # Spacing et taille
            if r.get("original_spacing"):
                f.write(f"Original spacing: {r['original_spacing']}\n")
                f.write(f"Original size : {r['original_size']}\n")
            if r.get("new_spacing"):
                f.write(f"New spacing : {r['new_spacing']}\n")
            if r.get("new_size"):
                f.write(f"New size : {r['new_size']}\n")
            
            # Normalisation
            if r.get("normalization_method"):
                f.write(f"Normalization : {r['normalization_method']}\n")
            if r.get("normalization_clip_bounds"):
                f.write(f"Clip bounds : {r['normalization_clip_bounds']}\n")
            
            # ROIs pour VascuSynth
            if dataset_name == "vascusynth":
                f.write(f"ROIs generated :\n")
                f.write(f"- Bifurcations : {'OUI'if r.get('bifurcation_mask_generated', False) else 'NON'}\n")
                f.write(f"- Neighborhood : {'OUI'if r.get('neighborhood_mask_generated', False) else 'NON'}\n")
            
            # Fichiers créés
            f.write(f"Files created :\n")
            for key, val in r.get("processed_files", {}).items():
                if isinstance(val, list):
                    for p in val:
                        f.write(f"- {Path(p).name}\n")
                elif val:
                    f.write(f"- {key}: {Path(val).name}\n")


    def _save_json(self, report: dict, filename: str) -> None:
        """Sauvegarde un rapport en JSON."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def _print_summary(self, results: list[dict], grouped: Dict[str, list[dict]]) -> None:
        """Affiche un résumé dans la console."""
        successful = [r for r in results if r["status"] == "success"]
        total_time = sum(r.get("time_seconds", 0) for r in successful)

        print("\n"+ "="* 80)
        print("PREPROCESSING COMPLETE")
        print("="* 80)
        print(f"Processed : {len(successful)}/{len(results)} patients")
        print(f"Total time : {total_time:.2f} s")
        print(f"Avg / pat : {total_time / max(len(successful), 1):.2f} s")
        
        print("\n By dataset:")
        for name, group in grouped.items():
            ok = [r for r in group if r["status"] == "success"]
            t = sum(r.get("time_seconds", 0) for r in ok)
            print(f"- {name:>10} : {len(ok):>3}/{len(group):<3} patients, {t:.2f} s")

        print(f"\n Reports : {self.output_dir}/[global|*]_preprocessing_report.[json|txt]")
        print("="* 80)