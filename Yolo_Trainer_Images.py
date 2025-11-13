from ultralytics import YOLO
import os
import yaml
import numpy as np

def descobrir_classes_reais():
    """Descobre as classes reais do dataset"""
    labels_path = r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8\train\labels'
    
    classes = set()
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            file_path = os.path.join(labels_path, label_file)
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            classes.add(class_id)
            except:
                continue
    
    return sorted(classes)

def criar_arquivo_data_yaml():
    """Cria o arquivo data.yaml necessário para o treinamento"""
    
    # Descobrir classes automaticamente
    classes_encontradas = descobrir_classes_reais()
    
    data_config = {
        'path': r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8',  # Caminho raiz
        'train': 'train',    # Pasta relativa ao path
        'val': 'valid',      # Pasta relativa ao path  
        'test': 'test',      # Pasta relativa ao path
        
        # Classes descobertas automaticamente
        'names': {class_id: f'estrutura_{class_id}' for class_id in classes_encontradas},
        'nc': len(classes_encontradas)  # Número de classes automático
    }
    
    # Salvar arquivo data.yaml
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print("✅ Arquivo data.yaml criado com sucesso!")
    print(f"🎯 Classes detectadas: {classes_encontradas}")
    return 'data.yaml'

def verificar_estrutura_pastas():
    """Verifica se a estrutura de pastas está correta"""
    
    base_path = r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8'
    pastas_necessarias = [
        'train/images',
        'train/labels', 
        'valid/images',
        'valid/labels',
        'test/images',
        'test/labels'
    ]
    
    print("🔍 Verificando estrutura de pastas...")
    for pasta in pastas_necessarias:
        caminho_completo = os.path.join(base_path, pasta)
        if os.path.exists(caminho_completo):
            # Contar arquivos
            if 'images' in pasta:
                num_files = len([f for f in os.listdir(caminho_completo) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✅ {pasta} - {num_files} imagens")
            else:
                num_files = len([f for f in os.listdir(caminho_completo) if f.endswith('.txt')])
                print(f"✅ {pasta} - {num_files} anotações")
        else:
            print(f"❌ Pasta não encontrada: {caminho_completo}")
            return False
    
    print("✅ Estrutura de pastas verificada com sucesso!")
    return True

def treinar_modelo():
    """Função principal para treinar o modelo YOLOv8"""
    
    # Verificar estrutura de pastas
    if not verificar_estrutura_pastas():
        print("❌ Erro na estrutura de pastas. Verifique os caminhos.")
        return None, None
    
    # Criar arquivo de configuração
    data_yaml = criar_arquivo_data_yaml()
    
    # Carregar modelo YOLOv8
    model = YOLO('yolov8n.pt')
    
    # Configurações de treinamento (otimizadas)
    config_treinamento = {
        'data': data_yaml,           # Arquivo de configuração
        'epochs': 350,               # Reduzido para teste
        'imgsz': 640,                # Tamanho da imagem
        'batch': 8,                  # Reduzido para evitar memory errors
        'patience': 15,              # Parada antecipada
        'device': 'cpu',             # 'cpu' ou 'cuda' para GPU
        'workers': 4,                # Threads de dataloader
        'project': 'runs/detect',    # Pasta do projeto
        'name': 'treinamento_mri',   # Nome do experimento
        'exist_ok': True,            # Sobrescrever se existir
        'verbose': True,             # Logs detalhados
        'save': True,                # Salvar checkpoints
        'amp': True,                 # Precisão mista (economia de memória)
    }
    
    # Iniciar treinamento
    print("🚀 Iniciando treinamento...")
    results = model.train(**config_treinamento)
    
    return model, results

def avaliar_modelo(model_path=None):
    """Avalia o modelo treinado - VERSÃO CORRIGIDA"""
    
    if model_path is None:
        # Tentar encontrar o modelo treinado automaticamente
        model_path = 'runs/detect/treinamento_mri/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado em: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Avaliar no conjunto de validação
    metrics = model.val()
    
    print("\n📊" + "="*50)
    print("📊 MÉTRICAS DE VALIDAÇÃO - DETALHADAS")
    print("="*50)
    
    # Métricas principais (são valores únicos)
    print(f"🎯 mAP@50-95: {metrics.box.map:.4f}")
    print(f"🎯 mAP@50: {metrics.box.map50:.4f}")
    print(f"🎯 mAP@75: {metrics.box.map75:.4f}")
    
    # CORREÇÃO: Precisão e Recall são arrays por classe
    if hasattr(metrics.box, 'p') and metrics.box.p is not None:
        if isinstance(metrics.box.p, (np.ndarray, list)):
            precision_media = np.mean(metrics.box.p)
            print(f"🎯 Precisão média: {precision_media:.4f}")
            # Mostrar precisão por classe
            for i, prec in enumerate(metrics.box.p):
                print(f"   Classe {i}: {prec:.4f}")
        else:
            print(f"🎯 Precisão: {metrics.box.p:.4f}")
    
    if hasattr(metrics.box, 'r') and metrics.box.r is not None:
        if isinstance(metrics.box.r, (np.ndarray, list)):
            recall_medio = np.mean(metrics.box.r)
            print(f"🎯 Recall médio: {recall_medio:.4f}")
            # Mostrar recall por classe
            for i, rec in enumerate(metrics.box.r):
                print(f"   Classe {i}: {rec:.4f}")
        else:
            print(f"🎯 Recall: {metrics.box.r:.4f}")
    
    # Informações adicionais
    print(f"📈 Número de classes: {getattr(metrics.box, 'nc', 'N/A')}")
    print(f"📈 Imagens processadas: {getattr(metrics, 'nt', 'N/A')}")
    
    return metrics

def fazer_predicoes(model_path=None):
    """Faz predições com o modelo treinado"""
    
    if model_path is None:
        model_path = 'runs/detect/treinamento_mri/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado em: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Fazer predição nas imagens de teste
    test_path = r'C:\Users\GuilhermeBragadoVale\Downloads\axial MRI.v2-release.yolov8\test\images'
    
    print("🎯 Fazendo predições nas imagens de teste...")
    results = model.predict(
        source=test_path,
        save=True,
        conf=0.5,      # Confiança mínima
        iou=0.5,       # IoU threshold
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    print(f"✅ Predições salvas em: runs/detect/treinamento_mri/predict/")
    
    # Mostrar estatísticas das predições
    if results:
        print(f"📊 {len(results)} imagens processadas")
        total_deteccoes = sum(len(result.boxes) for result in results if result.boxes is not None)
        print(f"📊 Total de detecções: {total_deteccoes}")
    
    return results

if __name__ == "__main__":
    # Executar treinamento
    modelo_treinado, resultados = treinar_modelo()
    
    if modelo_treinado is not None:
        # Avaliar modelo
        print("\n" + "="*60)
        print("AVALIAÇÃO DO MODELO TREINADO")
        print("="*60)
        metricas = avaliar_modelo()
        
        # Fazer predições
        print("\n" + "="*60)
        print("FAZENDO PREDIÇÕES NAS IMAGENS DE TESTE")
        print("="*60)
        predicoes = fazer_predicoes()
        
        print("\n🎉 TREINAMENTO E AVALIAÇÃO CONCLUÍDOS COM SUCESSO!")
        print("📍 Resultados salvos em: runs/detect/treinamento_mri/")
    else:
        print("❌ Treinamento não pode ser realizado.")