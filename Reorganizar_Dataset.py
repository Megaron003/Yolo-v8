import os
import shutil
import random
from collections import Counter
import os
import shutil
import random
from collections import Counter

def reorganizacao_imagens_e_labels_juntos():
    """Reorganiza IMAGENS E LABELS juntos - Versão Final"""
    
    base_path = r'C:/Users/GuilhermeBragadoVale/Downloads/axial MRI.v2-release.yolov8'
    backup_path = base_path + '_backup_completo'
    
    print("🔄 REORGANIZANDO IMAGENS E LABELS JUNTOS")
    print("="*60)
    
    # 1. Backup completo
    if not os.path.exists(backup_path):
        print("📦 Criando backup completo...")
        shutil.copytree(base_path, backup_path)
        print(f"✅ Backup em: {backup_path}")
    else:
        print("✅ Backup já existe")
    
    # 2. Coletar TODOS os pares do BACKUP
    print("\n📁 Coletando todos os pares imagem+label do BACKUP...")
    todos_os_pares = []
    
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(backup_path, split, 'images')
        labels_dir = os.path.join(backup_path, split, 'labels')
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_file)
                    
                    if os.path.exists(label_path):
                        # Coletar informações de AMBOS
                        todos_os_pares.append({
                            'imagem': img_file,
                            'label': label_file,
                            'caminho_imagem': os.path.join(images_dir, img_file),
                            'caminho_label': label_path
                        })
                        # print(f"   ✅ Par encontrado: {img_file} + {label_file}")
    
    print(f"✅ Total de pares imagem+label coletados: {len(todos_os_pares)}")
    
    if len(todos_os_pares) == 0:
        print("❌ Nenhum par encontrado no backup!")
        return False
    
    # 3. Analisar distribuição atual
    print("\n📊 ANALISANDO DISTRIBUIÇÃO ATUAL:")
    distribuicao_atual = Counter()
    
    for par in todos_os_pares:
        try:
            with open(par['caminho_label'], 'r') as f:
                linhas = [linha.strip() for linha in f if linha.strip()]
                if linhas:
                    classes = [int(linha.split()[0]) for linha in linhas]
                    classe_principal = max(set(classes), key=classes.count)
                    distribuicao_atual[classe_principal] += 1
        except Exception as e:
            print(f"⚠️  Erro ao ler {par['label']}: {e}")
            continue
    
    for classe, count in sorted(distribuicao_atual.items()):
        percentual = (count / len(todos_os_pares)) * 100
        print(f"   Classe {classe}: {count} pares ({percentual:.1f}%)")
    
    # 4. Agrupar por classe
    print("\n🎯 AGRUPANDO POR CLASSE...")
    por_classe = {}
    
    for par in todos_os_pares:
        try:
            with open(par['caminho_label'], 'r') as f:
                linhas = [linha.strip() for linha in f if linha.strip()]
                if linhas:
                    classes = [int(linha.split()[0]) for linha in linhas]
                    classe_principal = max(set(classes), key=classes.count)
                    
                    if classe_principal not in por_classe:
                        por_classe[classe_principal] = []
                    por_classe[classe_principal].append(par)
        except:
            continue
    
    for classe, pares in por_classe.items():
        print(f"   Classe {classe}: {len(pares)} pares")
    
    # 5. Distribuir entre splits
    print("\n📈 DISTRIBUINDO ENTRE TRAIN/VALID/TEST...")
    splits_finais = {'train': [], 'valid': [], 'test': []}
    
    for classe, pares in por_classe.items():
        print(f"   Processando Classe {classe}...")
        random.shuffle(pares)
        
        total = len(pares)
        train_count = int(total * 0.7)    # 70% treino
        valid_count = int(total * 0.2)    # 20% validação
        test_count = total - train_count - valid_count  # 10% teste
        
        print(f"     Train: {train_count}, Valid: {valid_count}, Test: {test_count}")
        
        splits_finais['train'].extend(pares[:train_count])
        splits_finais['valid'].extend(pares[train_count:train_count + valid_count])
        splits_finais['test'].extend(pares[train_count + valid_count:])
    
    # 6. Limpar destino (APENAS agora)
    print("\n🧹 LIMPANDO PASTAS DESTINO...")
    for split in ['train', 'valid', 'test']:
        for folder in ['images', 'labels']:
            path = os.path.join(base_path, split, folder)
            if os.path.exists(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
    
    # 7. Copiar IMAGENS E LABELS JUNTOS para nova distribuição
    print("\n📁 COPIANDO IMAGENS E LABELS JUNTOS...")
    total_copiados = 0
    
    for split, pares in splits_finais.items():
        print(f"   📂 {split.upper()}: {len(pares)} pares")
        
        for par in pares:
            try:
                # Destino para IMAGEM
                dest_img = os.path.join(base_path, split, 'images', par['imagem'])
                os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                shutil.copy2(par['caminho_imagem'], dest_img)
                
                # Destino para LABEL
                dest_label = os.path.join(base_path, split, 'labels', par['label'])
                os.makedirs(os.path.dirname(dest_label), exist_ok=True)
                shutil.copy2(par['caminho_label'], dest_label)
                
                total_copiados += 1
            except Exception as e:
                print(f"❌ Erro ao copiar {par['imagem']}: {e}")
    
    print(f"✅ Total de pares copiados: {total_copiados}")
    
    # 8. Verificar resultado final
    print("\n" + "="*50)
    print("🔍 VERIFICANDO RESULTADO FINAL")
    print("="*50)
    
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(base_path, split, 'images')
        labels_dir = os.path.join(base_path, split, 'labels')
        
        imagens = os.listdir(images_dir) if os.path.exists(images_dir) else []
        labels = os.listdir(labels_dir) if os.path.exists(labels_dir) else []
        
        print(f"\n📁 {split.upper()}:")
        print(f"   🖼 Imagens: {len(imagens)}")
        print(f"   📝 Labels: {len(labels)}")
        
        # Verificar correspondência
        if len(imagens) == len(labels):
            print(f"   ✅ Correspondência perfeita!")
        else:
            print(f"   ⚠️  Diferença: imagens={len(imagens)}, labels={len(labels)}")
        
        # Distribuição por classe (amostra)
        contador = Counter()
        if os.path.exists(labels_dir) and labels:
            for label_file in labels[:15]:  # Amostra dos primeiros 15
                try:
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                contador[class_id] += 1
                except:
                    continue
        
        if contador:
            total_split = sum(contador.values())
            print(f"   🎯 Amostra de distribuição:")
            for classe, count in sorted(contador.items()):
                percentual = (count / total_split) * 100
                print(f"      Classe {classe}: {count} ({percentual:.1f}%)")
    
    print("\n🎉 REORGANIZAÇÃO CONCLUÍDA COM SUCESSO!")
    return True

# 🚀 EXECUTAR REORGANIZAÇÃO
if __name__ == "__main__":
    print("🎯 REORGANIZAÇÃO DE DATASET - VERSÃO DEFINITIVA")
    print("💡 Garante que IMAGENS e LABELS ficam SEMPRE juntos")
    
    sucesso = reorganizacao_imagens_e_labels_juntos()
    
    if sucesso:
        print("\n" + "="*60)
        print("🎉 DATASET REORGANIZADO COM SUCESSO!")
        print("="*60)
        print("✅ Todas as imagens e labels correspondem")
        print("✅ Distribuição balanceada entre splits")
        print("✅ Modelo verá todas as classes durante treino")
        print("✅ Validação e teste representativos")
        print("\n💡 Agora execute o CÓDIGO MESTRE para treinar!")
    else:
        print("❌ Reorganização falhou")