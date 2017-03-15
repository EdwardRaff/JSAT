/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.classifiers.trees;

import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.evaluation.Accuracy;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.regression.evaluation.MeanSquaredError;
import jsat.regression.evaluation.RegressionScore;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * Mean Decrease in Accuracy (MDA) measures feature importance by applying the
 * classifier for each feature, and corruption one feature at a time as each
 * dataum its pushed through the tree. The importance of a feature is them
 * measured as the percent change in the target score when that feature was
 * corrupted. <br>
 * <br>
 * This approach is based off of Breiman, L. (2001). <i>Random forests</i>.
 * Machine Learning, 45(1), 5–32.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class MDA implements TreeFeatureImportanceInference
{
    
    private ClassificationScore cs_base = new Accuracy();
    private RegressionScore rs_base = new MeanSquaredError();

    @Override
    public <Type extends DataSet> double[] getImportanceStats(TreeLearner model, DataSet<Type> data)
    {
        double[] features = new double[data.getNumFeatures()];
        
        double baseScore;
        boolean percentIncrease;
        
        Random rand = RandomUtil.getRandom();
        if(data instanceof ClassificationDataSet)
        {
            ClassificationDataSet cds = (ClassificationDataSet) data;
            ClassificationScore cs = cs_base.clone();
            cs.prepare(cds.getPredicting());
            for(int i = 0; i < cds.getSampleSize(); i++)
            {
                DataPoint dp = cds.getDataPoint(i);
                cs.addResult(((Classifier)model).classify(dp), cds.getDataPointCategory(i), dp.getWeight());
            }
            baseScore = cs.getScore();
            percentIncrease = cs.lowerIsBetter();
            
            
            //for every feature
            for(int j  = 0; j < data.getNumFeatures(); j++)
            {
                cs.prepare(cds.getPredicting());
                
                for(int i = 0; i < cds.getSampleSize(); i++)
                {
                    DataPoint dp = cds.getDataPoint(i);
                    int true_label = cds.getDataPointCategory(i);
                    TreeNodeVisitor curNode = walkCorruptedPath(model, dp, j, rand);
                    
                    cs.addResult(curNode.localClassify(dp), true_label, dp.getWeight());
                }
                
                double newScore = cs.getScore();
                features[j] = percentIncrease ? (newScore-baseScore)/(baseScore+1e-3) : (baseScore-newScore)/(baseScore+1e-3);
            }
            
        }
        else if(data instanceof RegressionDataSet)
        {
            RegressionDataSet rds = (RegressionDataSet) data;
            RegressionScore rs = rs_base.clone();
            rs.prepare();
            for(int i = 0; i < rds.getSampleSize(); i++)
            {
                DataPoint dp = rds.getDataPoint(i);
                rs.addResult(((Regressor)model).regress(dp), rds.getTargetValue(i), dp.getWeight());
            }
            baseScore = rs.getScore();
            percentIncrease = rs.lowerIsBetter();
            
            
            //for every feature
            for(int j  = 0; j < data.getNumFeatures(); j++)
            {
                rs.prepare();
                
                for(int i = 0; i < rds.getSampleSize(); i++)
                {
                    DataPoint dp = rds.getDataPoint(i);
                    double true_label = rds.getTargetValue(i);
                    TreeNodeVisitor curNode = walkCorruptedPath(model, dp, j, rand);
                    
                    rs.addResult(curNode.localRegress(dp), true_label, dp.getWeight());
                }
                
                double newScore = rs.getScore();
                features[j] = percentIncrease ? (newScore-baseScore)/(baseScore+1e-3) : (baseScore-newScore)/(baseScore+1e-3);
            }
        }
        
        
        
        
        return features;
    }

    /**
     * walks the tree down to a leaf node, adding corruption for a specific feature
     * @param model the tree model to walk
     * @param dp the data point to push down the tree
     * @param j the feature index to corrupt
     * @param rand source of randomness
     * @return the leaf node 
     */
    private TreeNodeVisitor walkCorruptedPath(TreeLearner model, DataPoint dp, int j, Random rand)
    {
        TreeNodeVisitor curNode = model.getTreeNodeVisitor();
        while(!curNode.isLeaf())
        {
            int path = curNode.getPath(dp);
            int numChild = curNode.childrenCount();
            if(curNode.featuresUsed().contains(j))//corrupt the feature!
            {
                //this gets us a random OTHER path, wont be the same b/c we would need to wrap around 1 farther
                path = (path + rand.nextInt(numChild)) % numChild;
            }
            
            if(curNode.isPathDisabled(path))
                break;
            else
                curNode = curNode.getChild(path);
        }
        return curNode;
    }
    
}
