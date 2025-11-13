from ultralytics import YOLO
import os
import yaml
from collections import Counter

def analisar_dataset_completo():
    """Analisa TODOS os splits - Versão Corrigida"""
    
    base_path = r'C:/Users/GuilhermeBragadoVale/Downloads/axial MRI.v2-release.yolov8'
    
    print("🔍 ANALISANDO DATASET COMPLETO (TODOS OS SPLITS)")
    print("="*60)
    
    contador_geral = Counter()
    estatisticas = {}
    
    for split in ['train', 'valid', 'test']:
        labels_path = os.path.join(base_path, split, 'labels')
        images_path = os.path.join(base_path, split, 'images')
        
        print(f"\n📁 {split.upper()}:")
        
        # Verificar se pastas existem
        if not os.path.exists(labels_path):
            print(f"   ❌ Pasta labels não encontrada: {labels_path}")
            continue
        if not os.path.exists(images_path):
            print(f"   ❌ Pasta images não encontrada: {images_path}")
            continue
        
        # Contar arquivos
        imagens = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        
        print(f"   🖼 Imagens: {len(imagens)}")
        print(f"   📝 Labels: {len(labels)}")
        
        # Verificar correspondência
        imagens_sem_ext = {os.path.splitext(f)[0] for f in imagens}
        labels_sem_ext = {os.path.splitext(f)[0] for f in labels}
        
        sem_correspondencia = imagens_sem_ext - labels_sem_ext
        if sem_correspondencia:
            print(f"   ⚠️  Imagens sem labels: {len(sem_correspondencia)}")
        
        # Analisar distribuição de classes
        contador_split = Counter()
        total_anotacoes_split = 0
        
        for label_file in labels:
            try:
                with open(os.path.join(labels_path, label_file), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            contador_split[class_id] += 1
                            contador_geral[class_id] += 1
                            total_anotacoes_split += 1
            except Exception as e:
                print(f"   ⚠️  Erro em {label_file}: {e}")
                continue
        
        estatisticas[split] = {
            'imagens': len(imagens),
            'labels': len(labels),
            'distribuicao': dict(contador_split),
            'total_anotacoes': total_anotacoes_split
        }
        
        if contador_split:
            print(f"   🎯 Distribuição de classes:")
            for class_id, count in sorted(contador_split.items()):
                percentual = (count / total_anotacoes_split) * 100 if total_anotacoes_split > 0 else 0
                print(f"      Classe {class_id}: {count} anotações ({percentual:.1f}%)")
    
    # Estatísticas gerais
    print("\n" + "="*60)
    print("📊 ESTATÍSTICAS GERAIS:")
    print("="*60)
    
    total_imagens = sum(estatisticas[split]['imagens'] for split in estatisticas)
    total_anotacoes = sum(estatisticas[split]['total_anotacoes'] for split in estatisticas)
    
    print(f"📈 TOTAIS:")
    print(f"   🖼 Imagens: {total_imagens}")
    print(f"   📝 Anotações: {total_anotacoes}")
    print(f"   🎯 Classes detectadas: {len(contador_geral)}")
    
    if contador_geral:
        print(f"\n🎯 DISTRIBUIÇÃO GERAL:")
        for class_id, count in sorted(contador_geral.items()):
            percentual = (count / total_anotacoes) * 100
            print(f"   Classe {class_id}: {count} anotações ({percentual:.1f}%)")
    
    return estatisticas, contador_geral

def criar_data_yaml_inteligente(contador_geral):
    """Cria data.yaml baseado na análise completa"""
    
    print("\n📝 CRIANDO data.yaml INTELIGENTE...")
    
    base_path = r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8'
    
    data_config = {
        'path': base_path,
        'train': 'train',
        'val': 'valid',
        'test': 'test',
        'names': {class_id: f'estrutura_{class_id}' for class_id in sorted(contador_geral.keys())},
        'nc': len(contador_geral)
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print("✅ data.yaml CRIADO!")
    print(f"🎯 Classes configuradas: {list(sorted(contador_geral.keys()))}")
    
    return 'data.yaml'

def treinar_modelo_otimizado():
    """Treinamento com configuração otimizada"""
    
    print("\n🚀 INICIANDO TREINAMENTO OTIMIZADO")
    print("="*50)
    
    # 1. Análise completa do dataset
    estatisticas, contador_geral = analisar_dataset_completo()
    
    if not contador_geral:
        print("❌ Nenhuma classe detectada!")
        return None, None
    
    # 2. Criar data.yaml
    data_yaml = criar_data_yaml_inteligente(contador_geral)
    
    # 3. Configuração baseada na análise
    num_classes = len(contador_geral)
    
    config_treinamento = {
        'data': data_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',
        'patience': 20,
        'project': 'runs/detect',
        'name': 'treinamento_otimizado',
        'verbose': True,
        'save': True,
        
        # Configurações otimizadas
        'cls': 0.7,
        'lr0': 0.001,
        'weight_decay': 0.001,
        'optimizer': 'AdamW',
        
        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,
    }
    
    # Ajustes para desbalanceamento
    if num_classes > 1:
        max_count = max(contador_geral.values())
        min_count = min(contador_geral.values())
        
        if max_count / min_count > 3:
            print("🎯 Aplicando configuração anti-desbalanceamento...")
            config_treinamento.update({
                'cls': 0.8,
                'lr0': 0.0005,
            })
    
    # 4. Treinar
    try:
        print("📦 Carregando modelo YOLOv8...")
        model = YOLO('yolov8n.pt')
        
        print("🔥 Iniciando treinamento...")
        results = model.train(**config_treinamento)
        
        print("✅ Treinamento concluído!")
        return model, results
        
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        return None, None

def avaliar_modelo_completo():
    """Avaliação completa do modelo"""
    
    print("\n📊 INICIANDO AVALIAÇÃO COMPLETA")
    
    model_path = 'runs/detect/treinamento_otimizado/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Validar
    metrics = model.val()
    
    print("\n🎯 RESULTADOS DA AVALIAÇÃO:")
    print("="*40)
    
    if hasattr(metrics, 'box'):
        print(f"📈 mAP@50-95: {getattr(metrics.box, 'map', 0):.4f}")
        print(f"📈 mAP@50: {getattr(metrics.box, 'map50', 0):.4f}")
        print(f"📈 mAP@75: {getattr(metrics.box, 'map75', 0):.4f}")
        
        # Precisão e Recall (com tratamento seguro)
        if hasattr(metrics.box, 'p') and metrics.box.p is not None:
            if hasattr(metrics.box.p, 'mean'):
                print(f"🎯 Precisão média: {metrics.box.p.mean():.4f}")
        if hasattr(metrics.box, 'r') and metrics.box.r is not None:
            if hasattr(metrics.box.r, 'mean'):
                print(f"🎯 Recall médio: {metrics.box.r.mean():.4f}")
    
    return metrics

def fazer_predicoes_avancadas():
    """Faz predições com análise detalhada"""
    
    model_path = 'runs/detect/treinamento_otimizado/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    test_path = r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8\test\images'
    
    print(f"\n🎯 FAZENDO PREDIÇÕES EM: {test_path}")
    
    results = model.predict(
        source=test_path,
        save=True,
        conf=0.5,
        iou=0.5,
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    # Estatísticas das predições
    total_deteccoes = 0
    deteccoes_por_classe = Counter()
    
    for result in results:
        if result.boxes is not None:
            total_deteccoes += len(result.boxes)
            for cls in result.boxes.cls:
                deteccoes_por_classe[int(cls)] += 1
    
    print(f"\n📊 ESTATÍSTICAS DAS PREDIÇÕES:")
    print(f"   📈 Total de detecções: {total_deteccoes}")
    print(f"   🎯 Detecções por classe:")
    for classe, count in sorted(deteccoes_por_classe.items()):
        print(f"      Classe {classe}: {count}")
    
    print(f"\n✅ Predições salvas em: runs/detect/treinamento_otimizado/predict/")
    return results

# 🎯 PROGRAMA PRINCIPAL
if __name__ == "__main__":
    print("🎉 SISTEMA DE TREINAMENTO YOLOv8 - VERSÃO COMPLETA")
    print("="*60)
    
    # Treinar
    modelo, resultados = treinar_modelo_otimizado()
    
    if modelo is not None:
        # Avaliar
        metricas = avaliar_modelo_completo()
        
        # Fazer predições
        predicoes = fazer_predicoes_avancadas()
        
        print("\n🎉 PROCESSO COMPLETO CONCLUÍDO!")
        print("📍 Resultados em: runs/detect/treinamento_otimizado/")
    else:
        print("\n❌ Processo interrompido devido a erros")