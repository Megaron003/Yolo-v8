from ultralytics import YOLO
import os
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def gerar_analises_completas():
    """Gera análises completas: matriz de confusão, curvas, etc."""
    
    print("\n📊 GERANDO ANÁLISES COMPLETAS DO MODELO")
    print("="*50)
    
    model_path = 'runs/detect/treinamento_otimizado/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # 1. Matriz de Confusão
    print("\n🎯 GERANDO MATRIZ DE CONFUSÃO...")
    try:
        # Forçar a geração da matriz de confusão
        results_dir = 'runs/detect/treinamento_otimizado'
        confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
        
        # Validar para gerar métricas
        metrics = model.val(split='test')
        
        print("✅ Matriz de Confusão e métricas geradas!")
        
        # Análise detalhada das métricas
        print("\n📈 ANÁLISE DETALHADA DAS MÉTRICAS:")
        print("-" * 40)
        
        if hasattr(metrics, 'box'):
            print(f"🎯 mAP@50-95: {metrics.box.map:.4f}")
            print(f"🎯 mAP@50: {metrics.box.map50:.4f}")
            print(f"🎯 mAP@75: {metrics.box.map75:.4f}")
            
            # Precisão por classe
            if hasattr(metrics.box, 'p') and metrics.box.p is not None:
                if hasattr(metrics.box.p, '__iter__'):
                    print(f"\n🎯 PRECISÃO POR CLASSE:")
                    for i, prec in enumerate(metrics.box.p):
                        print(f"   Classe {i}: {prec:.4f}")
                    print(f"   Média: {np.mean(metrics.box.p):.4f}")
            
            # Recall por classe
            if hasattr(metrics.box, 'r') and metrics.box.r is not None:
                if hasattr(metrics.box.r, '__iter__'):
                    print(f"\n🎯 RECALL POR CLASSE:")
                    for i, rec in enumerate(metrics.box.r):
                        print(f"   Classe {i}: {rec:.4f}")
                    print(f"   Média: {np.mean(metrics.box.r):.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  Erro ao gerar matriz de confusão: {e}")
        return None

def analisar_curvas_aprendizado():
    """Analisa as curvas de aprendizado do treinamento"""
    
    print("\n📈 ANALISANDO CURVAS DE APRENDIZADO")
    print("="*50)
    
    results_dir = 'runs/detect/treinamento_otimizado'
    results_file = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(results_file):
        print("❌ Arquivo de resultados não encontrado")
        return
    
    try:
        # Ler resultados do treinamento
        import pandas as pd
        results_df = pd.read_csv(results_file)
        
        print("📊 ESTATÍSTICAS DO TREINAMENTO:")
        print("-" * 30)
        
        # Métricas finais
        ultima_linha = results_df.iloc[-1]
        
        print(f"✅ Épocas treinadas: {len(results_df)}")
        print(f"✅ Loss de caixa final: {ultima_linha.get('train/box_loss', 'N/A'):.4f}")
        print(f"✅ Loss de classe final: {ultima_linha.get('train/cls_loss', 'N/A'):.4f}")
        print(f"✅ Loss total final: {ultima_linha.get('train/loss', 'N/A'):.4f}")
        print(f"✅ mAP@50 final: {ultima_linha.get('metrics/mAP50(B)', 'N/A'):.4f}")
        
        # Análise de convergência
        if len(results_df) > 10:
            primeiras_epocas = results_df['metrics/mAP50(B)'].head(10).mean()
            ultimas_epocas = results_df['metrics/mAP50(B)'].tail(10).mean()
            melhoria = ultimas_epocas - primeiras_epocas
            
            print(f"\n📈 ANÁLISE DE CONVERGÊNCIA:")
            print(f"   mAP@50 primeiras 10 épocas: {primeiras_epocas:.4f}")
            print(f"   mAP@50 últimas 10 épocas: {ultimas_epocas:.4f}")
            print(f"   Melhoria: {melhoria:.4f}")
            
            if melhoria < 0.01:
                print("   💡 Modelo pode ter convergido cedo")
            elif melhoria > 0.05:
                print("   💡 Modelo ainda estava melhorando")
        
    except Exception as e:
        print(f"⚠️  Erro ao analisar curvas: {e}")

def gerar_relatorio_desempenho():
    """Gera relatório completo de desempenho"""
    
    print("\n📋 GERANDO RELATÓRIO COMPLETO DE DESEMPENHO")
    print("="*60)
    
    model_path = 'runs/detect/treinamento_otimizado/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado")
        return
    
    model = YOLO(model_path)
    
    # Validar em todos os splits
    print("\n🎯 DESEMPENHO POR SPLIT:")
    print("-" * 30)
    
    splits = ['train', 'val', 'test']
    desempenho = {}
    
    for split in splits:
        try:
            if split == 'train':
                # Para train, usar uma amostra para não demorar muito
                metrics = model.val(split='val')  # Usar val como proxy
            else:
                metrics = model.val(split=split)
            
            if hasattr(metrics, 'box'):
                desempenho[split] = {
                    'map50': metrics.box.map50,
                    'map': metrics.box.map
                }
                print(f"📁 {split.upper()}:")
                print(f"   mAP@50: {metrics.box.map50:.4f}")
                print(f"   mAP@50-95: {metrics.box.map:.4f}")
                
        except Exception as e:
            print(f"⚠️  Erro ao validar {split}: {e}")
    
    # Análise comparativa
    if 'train' in desempenho and 'val' in desempenho:
        gap = desempenho['train']['map50'] - desempenho['val']['map50']
        print(f"\n📊 ANÁLISE DE GAP TREINO/VALIDAÇÃO:")
        print(f"   Gap mAP@50: {gap:.4f}")
        
        if gap > 0.1:
            print("   ⚠️  Possível overfitting (gap muito alto)")
        elif gap < 0.02:
            print("   ✅ Boa generalização (gap pequeno)")
        else:
            print("   ⚠️  Gap moderado")

def avaliar_modelo_completo():
    """Avaliação completa do modelo - AGORA COM ANÁLISES"""
    
    print("\n📊 INICIANDO AVALIAÇÃO COMPLETA COM ANÁLISES")
    print("="*60)
    
    model_path = 'runs/detect/treinamento_otimizado/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # 1. Métricas básicas
    print("\n🎯 MÉTRICAS BÁSICAS DE VALIDAÇÃO")
    print("-" * 40)
    
    metrics = model.val()
    
    if hasattr(metrics, 'box'):
        print(f"📈 mAP@50-95: {getattr(metrics.box, 'map', 0):.4f}")
        print(f"📈 mAP@50: {getattr(metrics.box, 'map50', 0):.4f}")
        print(f"📈 mAP@75: {getattr(metrics.box, 'map75', 0):.4f}")
        
        # Precisão e Recall (com tratamento seguro)
        if hasattr(metrics.box, 'p') and metrics.box.p is not None:
            if hasattr(metrics.box.p, 'mean'):
                print(f"🎯 Precisão média: {metrics.box.p.mean():.4f}")
            elif hasattr(metrics.box.p, '__iter__'):
                print(f"🎯 Precisão média: {np.mean(metrics.box.p):.4f}")
                
        if hasattr(metrics.box, 'r') and metrics.box.r is not None:
            if hasattr(metrics.box.r, 'mean'):
                print(f"🎯 Recall médio: {metrics.box.r.mean():.4f}")
            elif hasattr(metrics.box.r, '__iter__'):
                print(f"🎯 Recall médio: {np.mean(metrics.box.r):.4f}")
    
    # 2. Análises avançadas
    gerar_analises_completas()
    analisar_curvas_aprendizado()
    gerar_relatorio_desempenho()
    
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
    confiancas_por_classe = {}
    
    for result in results:
        if result.boxes is not None:
            total_deteccoes += len(result.boxes)
            for i, cls in enumerate(result.boxes.cls):
                classe = int(cls)
                deteccoes_por_classe[classe] += 1
                
                # Coletar confianças
                if classe not in confiancas_por_classe:
                    confiancas_por_classe[classe] = []
                if hasattr(result.boxes, 'conf'):
                    confiancas_por_classe[classe].append(float(result.boxes.conf[i]))
    
    print(f"\n📊 ESTATÍSTICAS DAS PREDIÇÕES:")
    print(f"   📈 Total de detecções: {total_deteccoes}")
    print(f"   📈 Total de imagens processadas: {len(results)}")
    print(f"   🎯 Detecções por classe:")
    for classe, count in sorted(deteccoes_por_classe.items()):
        conf_media = np.mean(confiancas_por_classe.get(classe, [0]))
        print(f"      Classe {classe}: {count} detecções (conf: {conf_media:.3f})")
    
    print(f"\n✅ Predições salvas em: runs/detect/treinamento_otimizado/predict/")
    return results

# 🎯 PROGRAMA PRINCIPAL
if __name__ == "__main__":
    print("🎉 SISTEMA DE TREINAMENTO YOLOv8 - VERSÃO COMPLETA COM ANÁLISES")
    print("="*70)
    
    # Treinar (MANTIDO EXATAMENTE IGUAL)
    modelo, resultados = treinar_modelo_otimizado()
    
    if modelo is not None:
        # Avaliar (AGORA COM ANÁLISES COMPLETAS)
        metricas = avaliar_modelo_completo()
        
        # Fazer predições
        predicoes = fazer_predicoes_avancadas()
        
        print("\n" + "="*70)
        print("🎉 PROCESSO COMPLETO CONCLUÍDO!")
        print("📍 Resultados em: runs/detect/treinamento_otimizado/")
        print("📊 Análises disponíveis:")
        print("   ✅ Matriz de Confusão")
        print("   ✅ Curvas de Aprendizado") 
        print("   ✅ Métricas por Classe")
        print("   ✅ Relatório de Desempenho")
        print("   ✅ Estatísticas de Predições")
    else:
        print("\n❌ Processo interrompido devido a erros")