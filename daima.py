import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import re
import os
from collections import defaultdict
warnings.filterwarnings('ignore')#屏蔽警告信息

np.random.seed(42)#设置随机种子
tf.random.set_seed(42)# 设置随机种子

class BindingSiteInhibitorDataset:#定义类
    
    def __init__(self, interaction_threshold=0.3):#设置相互作用阈值 (30%)
        
        self.interaction_threshold = interaction_threshold #储存阈值到属性
        
        self.aa_properties = {
            'ALA': {'hydropathy': 1.8, 'pKa': 6.02, 'volume': 67, 'polarity': 0, 'charge': 0, 'hydrophobicity': 1.8},
            'ARG': {'hydropathy': -4.5, 'pKa': 10.76, 'volume': 148, 'polarity': 1, 'charge': 1, 'hydrophobicity': 4.5},
            'ASN': {'hydropathy': -3.5, 'pKa': 5.41, 'volume': 96, 'polarity': 1, 'charge': 0, 'hydrophobicity': 3.5},
            'ASP': {'hydropathy': -3.5, 'pKa': 2.77, 'volume': 91, 'polarity': 1, 'charge': -1, 'hydrophobicity': 3.5},
            'CYS': {'hydropathy': 2.5, 'pKa': 5.07, 'volume': 86, 'polarity': 0, 'charge': 0, 'hydrophobicity': 2.5},
            'GLN': {'hydropathy': -3.5, 'pKa': 5.65, 'volume': 114, 'polarity': 1, 'charge': 0, 'hydrophobicity': 3.5},
            'GLU': {'hydropathy': -3.5, 'pKa': 3.22, 'volume': 109, 'polarity': 1, 'charge': -1, 'hydrophobicity': 3.5},
            'GLY': {'hydropathy': -0.4, 'pKa': 5.97, 'volume': 48, 'polarity': 0, 'charge': 0, 'hydrophobicity': 0.4},
            'HIS': {'hydropathy': -3.2, 'pKa': 7.59, 'volume': 118, 'polarity': 1, 'charge': 0.5, 'hydrophobicity': 3.2},
            'ILE': {'hydropathy': 4.5, 'pKa': 6.02, 'volume': 124, 'polarity': 0, 'charge': 0, 'hydrophobicity': 4.5},
            'LEU': {'hydropathy': 3.8, 'pKa': 5.98, 'volume': 124, 'polarity': 0, 'charge': 0, 'hydrophobicity': 3.8},
            'LYS': {'hydropathy': -3.9, 'pKa': 9.74, 'volume': 135, 'polarity': 1, 'charge': 1, 'hydrophobicity': 3.9},
            'MET': {'hydropathy': 1.9, 'pKa': 5.75, 'volume': 114, 'polarity': 0, 'charge': 0, 'hydrophobicity': 1.9},
            'PHE': {'hydropathy': 2.8, 'pKa': 5.48, 'volume': 135, 'polarity': 0, 'charge': 0, 'hydrophobicity': 2.8},
            'PRO': {'hydropathy': -1.6, 'pKa': 6.30, 'volume': 90, 'polarity': 0, 'charge': 0, 'hydrophobicity': 1.6},
            'SER': {'hydropathy': -0.8, 'pKa': 5.68, 'volume': 73, 'polarity': 1, 'charge': 0, 'hydrophobicity': 0.8},
            'THR': {'hydropathy': -0.7, 'pKa': 5.60, 'volume': 93, 'polarity': 1, 'charge': 0, 'hydrophobicity': 0.7},
            'TRP': {'hydropathy': -0.9, 'pKa': 5.89, 'volume': 163, 'polarity': 0, 'charge': 0, 'hydrophobicity': 0.9},
            'TYR': {'hydropathy': -1.3, 'pKa': 5.66, 'volume': 141, 'polarity': 1, 'charge': 0, 'hydrophobicity': 1.3},
            'VAL': {'hydropathy': 4.2, 'pKa': 5.97, 'volume': 105, 'polarity': 0, 'charge': 0, 'hydrophobicity': 1.3}
        }      # 定义字典包含氨基酸的物理化学属性
        
        self.calc = Calculator(descriptors, ignore_3D=True)# 初始化Mordred计算器，只计算2D分子描述符
        
        from sklearn.preprocessing import LabelEncoder #从sklearn导入label类，用于氨基酸编码
        self.aa_encoder = LabelEncoder() #创建对象
        self.aa_encoder.fit(list(self.aa_properties.keys()))  # 对氨基酸进行编码
        
        self.amino_acid_columns = None 
        self.all_amino_acids = None   
        
        self.aa_name_mapping = {
            'ala': 'ALA', 'arg': 'ARG', 'asn': 'ASN', 'asp': 'ASP', 'cys': 'CYS',
            'gln': 'GLN', 'glu': 'GLU', 'gly': 'GLY', 'his': 'HIS', 'ile': 'ILE',
            'leu': 'LEU', 'lys': 'LYS', 'met': 'MET', 'phe': 'PHE', 'pro': 'PRO',
            'ser': 'SER', 'thr': 'THR', 'trp': 'TRP', 'tyr': 'TYR', 'val': 'VAL',
            'Ala': 'ALA', 'Arg': 'ARG', 'Asn': 'ASN', 'Asp': 'ASP', 'Cys': 'CYS',
            'Gln': 'GLN', 'Glu': 'GLU', 'Gly': 'GLY', 'His': 'HIS', 'Ile': 'ILE',
            'Leu': 'LEU', 'Lys': 'LYS', 'Met': 'MET', 'Phe': 'PHE', 'Pro': 'PRO',
            'Ser': 'SER', 'Thr': 'THR', 'Trp': 'TRP', 'Tyr': 'TYR', 'Val': 'VAL'
        }        # 氨基酸名称映射
    
    def load_excel_data(self, file_path='Ligand.xlsx'):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件 {file_path} 不存在，请确保文件路径正确")#检查文件是否存在
            
            df = pd.read_excel(file_path)
            print(f"从 {file_path} 成功加载数据，形状: {df.shape}")#加载文件
            

            smiles_col = df.columns[0]
            ic50_col = df.columns[-1]
            aa_columns = df.columns[1:-1]
            self.amino_acid_columns = list(aa_columns)
            self.all_amino_acids = self.amino_acid_columns.copy()  # 获取序列信息，第一列是SMILES，最后一列是IC50，中间是氨基酸
            
            print(f"SMILES列: {smiles_col}")
            print(f"氨基酸相互作用列 ({len(aa_columns)}个): {list(aa_columns)}")
            print(f"活性列: {ic50_col}")
            print(f"使用相互作用阈值: {self.interaction_threshold} (30%)")
            
            data = [] #创建一个空的数据表，存储处理后的数据
            
            for idx, row in df.iterrows(): #遍历每一行数据
                smiles = str(row[smiles_col]).strip() #提取当前行的Smiles数据，并去除空格活多余字符
                ic50 = row[ic50_col] #提取当前行的IC50
                
                if pd.isna(smiles) or smiles == '' or smiles == 'nan':
                    continue #检查当前行，若smiles为空则跳过或输出nan
                if pd.isna(ic50):
                    continue  #检查当前行，若IC50为空则跳过
                
                if ic50 > 0: 
                    activity_binary = 1
                else:  
                    activity_binary = 0 
                
                aa_interactions = [] #创建空的数据表用以存储当前分子与哪些氨基酸发生了相互作用
                interaction_vector = [] #存储二进制向量，表示每个氨基酸是否与分子发生了相互作用
                for aa_col in aa_columns: #遍历所有氨基酸列
                    aa_name = str(aa_col).strip() 
                    interaction = row[aa_col] #若该值不为NaN且大于设定的阈值，则表示该氨基酸与分子有相互作用
                    
                    if not pd.isna(interaction) and interaction > self.interaction_threshold:
                        aa_interactions.append(aa_name)
                        interaction_vector.append(1) #若具备相互作用则添加1
                    else:
                        interaction_vector.append(0) #若不具备相互作用则添加0
                
                data.append({
                    'smiles': smiles, # 分子SMILES
                    'amino_acids': aa_interactions, # 相互作用的氨基酸列表
                    'interaction_vector': interaction_vector, # 相互作用二进制向量
                    'activity': activity_binary, # 二进制活性标签
                    'original_ic50': ic50,  # 原始IC50值
                    'raw_interaction_values': [row[aa_col] for aa_col in aa_columns] # 所有原始相互作用值
                })
            
            print(f"从Excel文件成功加载 {len(data)} 个样本")
            print(f"平均相互作用氨基酸数量: {np.mean([len(d['amino_acids']) for d in data]):.2f}")
            
            return data
            
        except Exception as e:
            print(f"从Excel文件加载数据时出错: {e}")
            return [] #异常值处理
    
    def extract_aa_features(self, aa_list): #提取氨基酸相关的特征
        if not self.all_amino_acids:
            return np.zeros(50)  # 如果未定义返回50维的零向量
        
        interaction_vector = np.zeros(len(self.all_amino_acids))#初始化一个零向量，长度与self.all_amino_acids的氨基酸数量相同
        for i, aa in enumerate(self.all_amino_acids): #遍历self.all_amino_acids中的每一个氨基酸
            if aa in aa_list:
                interaction_vector[i] = 1        # 如果氨基酸在输入列表中，对应位置设置为1
        
        num_interactions = len(aa_list) #相互作用的氨基酸总数
        interaction_density = num_interactions / len(self.all_amino_acids) if self.all_amino_acids else 0 #相互作用密度（相互作用数/总氨基酸数）
        
        features = np.concatenate([
            interaction_vector,
            [num_interactions, interaction_density]
        ])  #将 interaction_vector（氨基酸的二进制相互作用向量）与 num_interactions 和 interaction_density 拼接在一起，形成一个完整的特征向量 features

        
        return features #返回这个特征向量
    
    def get_inhibitor_features(self, smiles):#提取小分子相关特征
        try:
            mol = Chem.MolFromSmiles(smiles)#使用函数将smiles字符串转化为分子对象mol
            if mol is not None:#检查是否成功解析分子
                descriptors_df = self.calc.pandas([mol])#使用x计算分子描述符并保存结果
                numeric_cols = descriptors_df.select_dtypes(include=[np.number]).columns
                descriptors_values = descriptors_df[numeric_cols].values[0]
                descriptors_values = np.nan_to_num(descriptors_values)#提取第一个分子的描述符数据，并使用函数将缺失值替换为0
                
                max_descriptors = 500 # 限制特征维度，避免过大
                if len(descriptors_values) > max_descriptors:
                    descriptors_values = descriptors_values[:max_descriptors]#超过500就截断
                elif len(descriptors_values) < max_descriptors:
                    descriptors_values = np.pad(descriptors_values, (0, max_descriptors - len(descriptors_values)), 'constant')#不到500用0填充
                
                return descriptors_values#返回最终的分子特征向量
            else:
                print(f"无法解析SMILES: {smiles}")
                return np.zeros(500)#若无法解析SMILES返回500维的零向量
        except Exception as e:
            print(f"计算抑制剂 {smiles} 特征时出错: {e}")
            return np.zeros(500) #若计算抑制剂特征时发生任何异常捕获该异常并打印错误信息，然后返回一个500维的零向量
    
    def build_dataset(self, data=None, use_excel=True, test_size=0.2):#准备数据集

        if data is None:
            if use_excel:
                data = self.load_excel_data()
            else:
                print("错误: 没有提供数据且未使用Excel数据")
                return None
        #检查数据，采用data或excel数据
        if not data:
            print("错误: 没有数据可用")
            return None
        
        smiles_list = list(set([d['smiles'] for d in data]))#提取所有smiles字符串并去重
        train_smiles, test_smiles = train_test_split(smiles_list, test_size=test_size, random_state=42)#分割数据集和测试集
        
        train_data = [d for d in data if d['smiles'] in train_smiles]
        test_data = [d for d in data if d['smiles'] in test_smiles]        # 分割数据集
        
        print(f"训练集: {len(train_data)} 个样本")
        print(f"测试集: {len(test_data)} 个样本")
        
        x_train, y_interaction_train, y_activity_train = self._build_features(train_data)
        x_test, y_interaction_test, y_activity_test = self._build_features(test_data)
        
        return (x_train, x_test, y_interaction_train, y_interaction_test, 
                y_activity_train, y_activity_test, train_data, test_data)        # 构建特征x_train训练集特征，x_test测试集，y_intx_train训练集相互作用标签，y_intx_test测试集，y_actx_train训练集活性标签，y_actx_test测试集，train_data训练集原始数据，test_data测试集原始数据
    
    def _build_features(self, data):
 
        x_data = [] #储存特征向量
        y_interaction = [] #相互作用标签
        y_activity = [] #活性标签
        
        for sample in data:#遍历data的所有样本
            try:#处理出现错误跳过当前样本
                
                inhibitor_features = self.get_inhibitor_features(sample['smiles']) # 抑制剂特征
                
                aa_features = self.extract_aa_features(sample['amino_acids']) # 氨基酸相互作用特征
                
                combined_features = np.concatenate([inhibitor_features, aa_features])# 合并特征
                x_data.append(combined_features)#合并特征添加到x_data
                
                y_interaction.append(sample['interaction_vector'])#将相互作用向量添加到y_interaction
                y_activity.append(sample['activity'])# 活性数据添加到y_activity
                
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue #捕获异常，出现错误打印错误信息并跳过当前样本
        
        return np.array(x_data), np.array(y_interaction), np.array(y_activity)

class InteractionBasedActivityModel:#定义类

    
    def __init__(self, input_dim, num_amino_acids, interaction_threshold=0.3):#构造函数
        self.input_dim = input_dim #输入特征维度
        self.num_amino_acids = num_amino_acids #氨基酸数量
        self.interaction_threshold = interaction_threshold #相互作用阈值
        self.model = None #储存构建的模型
        self.scaler = StandardScaler() #特征标准化
        self.history = None #储存训练历史
    
    def build_model(self):#构建模型方法

        inputs = tf.keras.layers.Input(shape=(self.input_dim,)) #定义模型的输入形状，与特征维度匹配
        
        # 共享特征提取层
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x) #防止过拟合，提高泛化能力
        x = tf.keras.layers.Dense(128, activation='relu')(x) #逐渐减少神经元数量，提取高层次特征
        
        interaction_output = tf.keras.layers.Dense(
            self.num_amino_acids, #每个氨基酸一个输出
            activation='sigmoid', #独立概率
            name='interaction'
        )(x)    # 预测与每个氨基酸的相互作用概率（多标签分类）
        
        combined = tf.keras.layers.Concatenate()([x, interaction_output])#合并提取的特征x和interaction_output
        
        activity_branch = tf.keras.layers.Dense(64, activation='relu')(combined)
        activity_branch = tf.keras.layers.Dropout(0.2)(activity_branch)
        activity_branch = tf.keras.layers.Dense(32, activation='relu')(activity_branch)
        activity_output = tf.keras.layers.Dense(1, activation='sigmoid', name='activity')(activity_branch)#二分类输出（有活性/无活性）
        
        self.model = tf.keras.models.Model(
            inputs=inputs, 
            outputs=[interaction_output, activity_output]
        )#定义模型输出包含两部分相互作用概率和预测样本的生物活性

        #编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),#指定优化器为Adam
            loss={
                'interaction': 'binary_crossentropy',#多标签分类
                'activity': 'binary_crossentropy'#二分类
            },
            loss_weights={'interaction': 0.4, 'activity': 0.6},#辅助任务权重较低，主任务权重较高
            metrics={
                'interaction': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                'activity': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            }
        )
        
        print(f"基于相互作用的活性预测模型构建完成 (阈值: {self.interaction_threshold})")
        self.model.summary()
    
    def train(self, x_train, y_interaction_train, y_activity_train, 
              x_val, y_interaction_val, y_activity_val, epochs=100, batch_size=32):

        # 对训练和验证数据进行标准化。
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)
        
        # 回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),#在训练过程中，如果验证集的损失（val_loss）连续 patience 20个验证损失没有降低，则停止训练。并且 restore_best_weights=True 确保恢复训练过程中的最佳权重，避免过拟合
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)#如果验证损失在连续 patience 10个 验证损失没有降低没有改善，则降低学习率（factor=0.5）
        ]
        
        # 训练
        self.history = self.model.fit(
            x_train_scaled,#标准化后的训练数据特征
            {
                'interaction': y_interaction_train,#训练数据中的相互作用标签
                'activity': y_activity_train#训练数据中的活性数据标签
            },
            validation_data=(
                x_val_scaled,
                {
                    'interaction': y_interaction_val,
                    'activity': y_activity_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
    #预测
    def predict(self, x):

        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)
        
    #评估模型
    def evaluate(self, x_test, y_interaction_test, y_activity_test):
  
        x_test_scaled = self.scaler.transform(x_test)
        results = self.model.evaluate(
            x_test_scaled,
            {
                'interaction': y_interaction_test,
                'activity': y_activity_test
            },
            verbose=0
        )
        
        # 解析结果
        metrics = {}
        if len(results) >= 7:  # 总损失 + 2个任务的损失和指标
            metrics = {
                'total_loss': results[0],#总损失
                'interaction_loss': results[1],#相互作用损失
                'activity_loss': results[2],#活性损失
                'interaction_accuracy': results[3],#相互作用准确率
                'interaction_precision': results[4],#精确度
                'interaction_recall': results[5],#召回率
                'activity_accuracy': results[6],#活性准确率
                'activity_precision': results[7],#精确度
                'activity_recall': results[8]#召回率
            }
        
        return metrics
        
   #预测与阈值应用 
    def predict_with_threshold(self, x, threshold=None):

        if threshold is None:
            threshold = self.interaction_threshold
        
        predictions = self.predict(x)
        interaction_probs = predictions[0]#模型对interaction的预测概率
        activity_probs = predictions[1]#模型对activity的预测概率
        
        # 应用阈值
        interaction_predictions = (interaction_probs > threshold).astype(int)
        
        return interaction_probs, interaction_predictions, activity_probs

def predict_new_compound(model, dataset, smiles, threshold=None):

    if threshold is None:
        threshold = model.interaction_threshold#设置阈值
    
    try:
        print(f"\n预测化合物: {smiles}")
        print("="*50)
        print(f"使用相互作用阈值: {threshold} (30%)")
        
        # 获取特征
        inhibitor_features = dataset.get_inhibitor_features(smiles)#提取smiles特征
        aa_features = dataset.extract_aa_features([]) #数据集氨基酸特征，无相互作用传入空列表
        
        combined_features = np.concatenate([inhibitor_features, aa_features])#合并化合物特征和氨基酸特征
        combined_features = combined_features.reshape(1, -1)#将组合后的特征调整为二维数组
        
        # 预测
        interaction_probs, interaction_pred, activity_prob = model.predict_with_threshold(combined_features, threshold)
        
        # 解析相互作用预测
        interaction_probs = interaction_probs[0]  # 取第一个样本
        interaction_pred = interaction_pred[0]  # 取第一个样本的预测结果
        activity_prob = activity_prob[0][0]  # 取第一个样本的活性概率
        
        print("氨基酸相互作用预测 (概率 > 30% 认为有相互作用):")
        print("-" * 50)
        
        interacting_aas = []
        high_prob_aas = []  # 高概率相互作用（>60%）
        
        for i, aa in enumerate(dataset.all_amino_acids):#遍历所有氨基酸与阈值比较
            prob = interaction_probs[i]
            if prob > threshold:
                interacting_aas.append((aa, prob))
                if prob > 0.6:  # 高概率
                    high_prob_aas.append((aa, prob))
                    print(f"✓✓ {aa}: 强相互作用 (概率: {prob:.3f})")
                elif prob > 0.3:  # 中等概率
                    print(f"✓  {aa}: 中等相互作用 (概率: {prob:.3f})")
            else:
                print(f"   {aa}: 无相互作用 (概率: {prob:.3f})")
        
        print(f"\n基于相互作用的活性预测:")
        print("-" * 30)
        
        # 分析相互作用模式对活性的影响
        num_interactions = len(interacting_aas)#计算与化合物发生相互作用的氨基酸数量
        interaction_strength = num_interactions / len(dataset.all_amino_acids) if dataset.all_amino_acids else 0#计算相互作用密度
        
        print(f"相互作用氨基酸数量 (>30%): {num_interactions}")
        print(f"强相互作用氨基酸数量 (>60%): {len(high_prob_aas)}")
        print(f"相互作用密度: {interaction_strength:.3f}")
        
        # 根据相互作用模式给出活性预测解释
        if activity_prob > 0.5:
            print(f"✓ 预测有活性 (概率: {activity_prob:.3f})")
            
            if num_interactions > 0:
                print("  关键相互作用氨基酸:")
                # 按概率排序显示前5个最重要的
                sorted_interactions = sorted(interacting_aas, key=lambda x: x[1], reverse=True)
                for aa, prob in sorted_interactions[:5]:
                    strength = "强" if prob > 0.6 else "中等"
                    print(f"  - {aa}: {strength}相互作用 (概率: {prob:.3f})")
                
                # 给出活性机制解释
                if len(high_prob_aas) >= 2:
                    print("  机制: 多个强相互作用表明良好的结合亲和力")
                elif num_interactions >= 3:
                    print("  机制: 多个中等相互作用提供稳定的结合")
                else:
                    print("  机制: 有限的相互作用可能影响活性强度")
            else:
                print("  注意: 未检测到显著的氨基酸相互作用")
        else:
            print(f"✗ 预测无活性 (概率: {activity_prob:.3f})")
            
            if num_interactions == 0:
                print("  原因: 未检测到显著的氨基酸相互作用")
            elif num_interactions < 2:
                print("  原因: 相互作用数量不足")
            else:
                print("  原因: 相互作用模式不足以产生活性")
        
        return {
            'interacting_amino_acids': [aa for aa, prob in interacting_aas],
            'interaction_probabilities': interaction_probs,
            'activity_probability': activity_prob,
            'num_interactions': num_interactions,
            'interaction_strength': interaction_strength,
            'strong_interactions': len(high_prob_aas)
        }
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        return None

def analyze_interaction_patterns(dataset, model, x_test, y_activity_test, threshold=None):

    if threshold is None:
        threshold = model.interaction_threshold
    
    print(f"\n分析相互作用模式与活性的关系 (阈值: {threshold}):")
    print("="*50)
    
    # 获取测试集预测
    interaction_probs, interaction_pred, activity_pred = model.predict_with_threshold(x_test, threshold)
    
    # 分析不同相互作用数量对应的活性概率
    interaction_counts = np.sum(interaction_pred, axis=1)
    
    results = []
    unique_counts = np.unique(interaction_counts)
    
    for count in unique_counts:
        mask = interaction_counts == count
        if np.sum(mask) > 0:
            avg_activity_prob = np.mean(activity_pred[mask])
            actual_activity_ratio = np.mean(y_activity_test[mask])
            results.append({
                'interaction_count': count,
                'sample_count': np.sum(mask),
                'avg_predicted_activity': avg_activity_prob,
                'actual_activity_ratio': actual_activity_ratio
            })
    
    # 显示结果
    print("相互作用数量 (>30%) vs 活性概率:")
    print("数量\t样本数\t预测活性\t实际活性比例")
    for result in sorted(results, key=lambda x: x['interaction_count']):
        print(f"{result['interaction_count']}\t{result['sample_count']}\t{result['avg_predicted_activity']:.3f}\t\t{result['actual_activity_ratio']:.3f}")
    
    # 分析关键氨基酸的相互作用频率
    print(f"\n关键氨基酸相互作用频率 (>30%阈值):")
    print("-" * 40)
    
    active_mask = y_activity_test == 1
    inactive_mask = y_activity_test == 0
    
    if np.sum(active_mask) > 0 and np.sum(inactive_mask) > 0:
        active_interactions = np.sum(interaction_pred[active_mask], axis=0)
        inactive_interactions = np.sum(interaction_pred[inactive_mask], axis=0)
        
        active_freq = active_interactions / np.sum(active_mask)
        inactive_freq = inactive_interactions / np.sum(inactive_mask)
        
        # 找出在有活性样本中频率更高的氨基酸
        important_aas = []
        for i, aa in enumerate(dataset.all_amino_acids):
            if active_freq[i] > inactive_freq[i] and active_freq[i] > 0.2:
                important_aas.append((aa, active_freq[i], inactive_freq[i]))
        
        # 按频率排序显示
        important_aas.sort(key=lambda x: x[1], reverse=True)
        
        print("氨基酸\t活性样本频率\t非活性样本频率")
        for aa, active_freq, inactive_freq in important_aas[:10]:  # 显示前10个
            print(f"{aa}\t{active_freq:.3f}\t\t{inactive_freq:.3f}")
    
    return results

def detailed_interaction_analysis(dataset, train_data, test_data):

    print("\n" + "="*60)
    print("详细相互作用分析")
    print("="*60)
    
    # 分析训练数据中的相互作用模式
    all_interactions = []
    for data in [train_data, test_data]:
        for sample in data:
            all_interactions.extend(sample['amino_acids'])
    
    # 计算每个氨基酸的出现频率
    aa_freq = {}
    for aa in dataset.all_amino_acids:
        freq = all_interactions.count(aa) / len(all_interactions) * 100 if all_interactions else 0
        aa_freq[aa] = freq
    
    print("氨基酸相互作用频率统计:")
    print("-" * 40)
    for aa, freq in sorted(aa_freq.items(), key=lambda x: x[1], reverse=True):
        if freq > 0:
            print(f"{aa}: {freq:.1f}%")
    
    # 分析活性与相互作用的关系
    active_samples = [d for d in train_data + test_data if d['activity'] == 1]
    inactive_samples = [d for d in train_data + test_data if d['activity'] == 0]
    
    if active_samples and inactive_samples:
        avg_active_interactions = np.mean([len(d['amino_acids']) for d in active_samples])
        avg_inactive_interactions = np.mean([len(d['amino_acids']) for d in inactive_samples])
        
        print(f"\n活性样本平均相互作用数: {avg_active_interactions:.2f}")
        print(f"非活性样本平均相互作用数: {avg_inactive_interactions:.2f}")
        
        if avg_active_interactions > avg_inactive_interactions:
            print("✓ 活性样本有更多的氨基酸相互作用")
        else:
            print("✗ 活性样本的相互作用数没有明显优势")

def main():

    print("基于氨基酸相互作用的活性预测模型 (30%阈值)")
    print("="*60)
    
    # 设置相互作用阈值
    INTERACTION_THRESHOLD = 0.3
    
    # 1. 加载数据
    print("步骤1: 加载数据")
    dataset = BindingSiteInhibitorDataset(interaction_threshold=INTERACTION_THRESHOLD)
    
    result = dataset.build_dataset(use_excel=True, test_size=0.2)
    if result is None:
        print("错误: 数据加载失败")
        return
    
    (x_train, x_test, y_interaction_train, y_interaction_test, 
     y_activity_train, y_activity_test, train_data, test_data) = result
    
    print(f"训练集特征维度: {x_train.shape}")
    print(f"测试集特征维度: {x_test.shape}")
    print(f"氨基酸数量: {len(dataset.all_amino_acids)}")
    
    # 2. 构建模型
    print("\n步骤2: 构建模型")
    model = InteractionBasedActivityModel(
        input_dim=x_train.shape[1],
        num_amino_acids=len(dataset.all_amino_acids),
        interaction_threshold=INTERACTION_THRESHOLD
    )
    model.build_model()
    
    # 3. 训练模型（使用部分训练数据作为验证集）
    x_train_final, x_val, y_interaction_train_final, y_interaction_val = train_test_split(
        x_train, y_interaction_train, test_size=0.2, random_state=42
    )
    y_activity_train_final, y_activity_val = train_test_split(
        y_activity_train, test_size=0.2, random_state=42
    )
    
    print("\n步骤3: 训练模型")
    model.train(
        x_train_final, y_interaction_train_final, y_activity_train_final,
        x_val, y_interaction_val, y_activity_val,
        epochs=100, batch_size=32
    )
    
    # 4. 评估模型
    print("\n步骤4: 评估模型")
    test_metrics = model.evaluate(x_test, y_interaction_test, y_activity_test)
    print("测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 5. 详细相互作用分析
    detailed_interaction_analysis(dataset, train_data, test_data)
    
    # 6. 分析相互作用模式
    analyze_interaction_patterns(dataset, model, x_test, y_activity_test)
    
    # 7. 预测新化合物
    print("\n步骤5: 新化合物预测示例")
    test_compounds = [
        "CCOC(=O)C1=CC=CC=C1",  # 苯甲酸乙酯
        "CCN(CC)CC",             # 三乙胺
        "CCOC(=O)C1=CC=C(C=C1)Cl",  # 氯苯甲酸乙酯
    ]
    
    for compound in test_compounds:
        result = predict_new_compound(model, dataset, compound)
        if result:
            print(f"\n预测摘要:")
            print(f"- 相互作用氨基酸 (>30%): {result['num_interactions']}个")
            print(f"- 强相互作用氨基酸 (>60%): {result['strong_interactions']}个")
            print(f"- 活性概率: {result['activity_probability']:.3f}")
            print(f"- 预测结果: {'有活性' if result['activity_probability'] > 0.5 else '无活性'}")
    
    # 8. 保存模型
    print("\n步骤6: 保存模型")
    model.model.save('interaction_based_activity_model_30threshold.h5')
    with open('dataset_info.pkl', 'wb') as f:
        pickle.dump({
            'amino_acids': dataset.all_amino_acids,
            'feature_dim': x_train.shape[1],
            'interaction_threshold': INTERACTION_THRESHOLD
        }, f)
    
    print("模型保存完成!")
    
    # 交互式预测
    print("\n" + "="*50)
    print("交互式预测界面 (30%阈值)")
    print("输入'quit'退出")
    
    while True:
        smiles = input("\n请输入化合物SMILES: ").strip()
        if smiles.lower() == 'quit':
            break
        if smiles:
            predict_new_compound(model, dataset, smiles)

if __name__ == "__main__":
    main()
