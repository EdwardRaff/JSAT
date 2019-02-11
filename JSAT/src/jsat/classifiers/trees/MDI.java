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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.utils.IntList;

/**
 * Determines the importance of features by measuring the decrease in impurity
 * caused by each feature used, weighted by the amount of data seen by the node
 * using the feature. <br>
 * This method only works for classification datasets as it uses the
 * {@link ImpurityScore} class, but may use any impurity measure supported.<br>
 * <br>
 * For more info, see:
 * <ul>
 * <li>Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013).
 * <i>Understanding variable importances in forests of randomized trees</i>. In
 * C. j. c. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. q. Weinberger
 * (Eds.), Advances in Neural Information Processing Systems 26 (pp. 431–439).
 * Retrieved from
 * <a href="http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/281.pdf">here</a></li>
 * <li>Breiman, L. (2002). Manual on setting up, using, and understanding random
 * forests v3.1. Statistics Department University of California Berkeley, CA,
 * USA.</li>
 * </ul>
 *
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class MDI implements TreeFeatureImportanceInference
{
    private ImpurityScore.ImpurityMeasure im;
    
    public MDI(ImpurityScore.ImpurityMeasure im)
    {
        this.im = im;
    }

    public MDI()
    {
        this(ImpurityScore.ImpurityMeasure.GINI);
    }
    

    @Override
    public <Type extends DataSet> double[] getImportanceStats(TreeLearner model, DataSet<Type> data)
    {
        double[] features = new double[data.getNumFeatures()];
        
        if(!(data instanceof ClassificationDataSet))
            throw new RuntimeException("MDI currently only supports classification datasets");
        
        List<DataPointPair<Integer>> allData = ((ClassificationDataSet)data).getAsDPPList();
        final int K = ((ClassificationDataSet)data).getClassSize();
        ImpurityScore score = new ImpurityScore(K, im);
        for(int i = 0; i < data.size(); i++)
            score.addPoint(data.getWeight(i), ((ClassificationDataSet)data).getDataPointCategory(i));
        
        visit(model.getTreeNodeVisitor(), score, (ClassificationDataSet) data, IntList.range(data.size()), features, score.getSumOfWeights(), K);
        
        return features;
    }
    
    private void visit(TreeNodeVisitor node, ImpurityScore score, ClassificationDataSet data, IntList subset, final double[] features , final double N, final int K)
    {
        if (node == null || node.isLeaf() )//invalid path or no split, so skip
            return;
        
        double curScore = score.getScore();
        double curN = score.getSumOfWeights();
        
        //working space to split data up into new subsets
        List<IntList> splitsData = new ArrayList<>(node.childrenCount());
        List<ImpurityScore> splitScores = new ArrayList<>(node.childrenCount());
        splitsData.add(subset);
        splitScores.add(score);
        for(int i = 0; i < node.childrenCount()-1; i++)
        {
            splitsData.add(new IntList());
            splitScores.add(new ImpurityScore(K, im));
        }
        
        //loop through and split up our data
        for(ListIterator<Integer> iter = subset.listIterator(); iter.hasNext();)
        {
	    int indx = iter.next();
            final int tc = data.getDataPointCategory(indx);
            DataPoint dp = data.getDataPoint(indx);
	    double w = data.getWeight(indx);
            int path = node.getPath(dp);
            if(path < 0)//NaN will cause -1
                score.removePoint(w, tc);
            else if(path > 0)//0 will be cur data and score obj, else we move to right location
            {
                score.removePoint(w, tc);
                splitScores.get(path).addPoint(w, tc);
                splitsData.get(path).add(indx);
                iter.remove();
            }
        }
        
        double chageInImp = curScore;
        for(ImpurityScore s : splitScores)
            chageInImp -= s.getScore()*(s.getSumOfWeights()/(1e-5+curN));
        
        
        Collection<Integer> featuresUsed = node.featuresUsed();
        for (int feature : featuresUsed)
            features[feature] += chageInImp*curN/N;

        //now visit our children
        for(int path = 0; path < splitScores.size(); path++)
            visit(node.getChild(path), splitScores.get(path), data, splitsData.get(path), features, N, K);
    }
    
}
