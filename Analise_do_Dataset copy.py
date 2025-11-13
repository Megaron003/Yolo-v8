# diagnostico_dataset.py - Arquivo separado
import os
from collections import Counter

def diagnostico_completo():
    """Script independente para analisar seu dataset"""
    
    base_path = r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8'
    
    print("🔍 DIAGNÓSTICO COMPLETO DO DATASET")
    print("="*50)
    
    for split in ['train', 'valid', 'test']:
        print(f"\n📁 {split.upper()}:")
        
        images_dir = os.path.join(base_path, split, 'images')
        labels_dir = os.path.join(base_path, split, 'labels')
        
        # Contar imagens
        imagens = [f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))] if os.path.exists(images_dir) else []
        labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] if os.path.exists(labels_dir) else []
        
        print(f"  🖼 Imagens: {len(imagens)}")
        print(f"  📝 Labels: {len(labels)}")
        
        # Analisar distribuição de classes
        if os.path.exists(labels_dir):
            contador = Counter()
            for label_file in labels[:100]:  # Analisa até 100 arquivos
                try:
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                contador[class_id] += 1
                except:
                    continue
            
            if contador:
                print(f"  🎯 Distribuição de classes:")
                for class_id, count in sorted(contador.items()):
                    print(f"     Classe {class_id}: {count} anotações")

if __name__ == "__main__":
    diagnostico_completo()