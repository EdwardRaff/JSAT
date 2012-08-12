
package jsat.graphing;

import java.awt.Color;
import java.awt.Graphics2D;
import jsat.classifiers.*;
import jsat.linear.DenseVector;

/**
 * A plotting tool to visualize the classification results on a 2 dimensional data set. 
 * @author Edward Raff
 */
public class ClassificationPlot extends CategoryPlot
{
    private Classifier classifier;
    private int resolution = 4;
    private boolean hardBoundaries = true;
    
    /**
     * Creates a new ClassificatoinPlot for visualizing some data set
     * @param dataSet the data set to plot
     * @param classifier the already trained classifier to determine which class
     * points belong to
     */
    public ClassificationPlot(ClassificationDataSet dataSet, Classifier classifier)
    {
        super(dataSet);
        this.classifier = classifier;
    }

    /**
     * Sets the plotting resolution for classification results. Points will be 
     * classified in a grid like manner. A higher resolution is faster, but will
     * look blockier. 
     * @param resolution the block resolution size. Values less than zero will be ignored
     */
    public void setResolution(int resolution)
    {
        if(resolution > 0)
            this.resolution = resolution;
        forceRedraw();
    }

    /**
     * The plotter can show the final decision class, or it can share and mix 
     * the colors to indicate the change in class probabilities. 
     * @param hardBoundaries <tt>true</tt> to only indicate the final decision, 
     * <tt>false</tt> to mix colors in proportion to the class probabilities. 
     */
    public void setHardBoundaries(boolean hardBoundaries)
    {
        if(this.hardBoundaries == hardBoundaries)
            return;
        this.hardBoundaries = hardBoundaries;
        forceRedraw();
    }

    /**
     * Returns true if hard boundaries are being drawn, false if colors are 
     * mixed according to class probability. 
     * @return true if hard boundaries are being drawn, false if colors are 
     * mixed according to class probability. 
     */
    public boolean isHardBoundaries()
    {
        return hardBoundaries;
    }
    
    /**
     * Returns the block size resolution used in plotting
     * @return the resolution used in plotting
     */
    public int getResolution()
    {
        return resolution;
    }

    
    @Override
    protected void drawPoints(Graphics2D g2, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        int[] noCatVals = new int[0];
        CategoricalData[] noCatData = new CategoricalData[0];
        double full = (imageWidth-PAD*2)/resolution*(imageHeight-PAD*2)/resolution;
        
        if(pp != null)
        {
            pp.setNote("Computing Boundry");
            pp.getjProgressBar().setMaximum((int)Math.ceil(full));
            pp.getjProgressBar().setIndeterminate(false);
        }
        
        
        for(int i = PAD+1; i < imageWidth-PAD; i+=resolution)
        {
            for(int j = PAD+1; j < imageHeight-PAD; j+=resolution)
            {
                CategoricalResults cr = classifier.classify(
                        new DataPoint(
                        DenseVector.toDenseVec(toXVal(i, imageWidth), toYVal(j, imageHeight)),
                        noCatVals, noCatData));
                if(cr.getProb(cr.mostLikely()) < 1e-17)
                    continue;
                
                if(hardBoundaries)
                {
                    int crClass = cr.mostLikely();
                    g2.setColor(getCategoryColor(crClass).brighter());
                }
                else
                {
                    float R=0, G=0, B=0;
                    for(int z = 0; z < cr.size(); z++)
                    {
                        Color c = getCategoryColor(z).brighter();
                        R += c.getRed()*cr.getProb(z);
                        G += c.getGreen()*cr.getProb(z);
                        B += c.getBlue()*cr.getProb(z);
                    }
                    
                    g2.setColor(new Color(R/256f, G/256f, B/256f));
                }
                g2.fillRect(i, j, resolution, resolution);
                
                if(pp != null)
                {
                    double cur = (imageWidth-PAD*2)*i/resolution+(j-PAD)/resolution;
                    pp.getjProgressBar().setValue((int)cur);
                }
            }
        }
        
        super.drawPoints(g2, imageWidth, imageHeight, pp);
    }
    
    
    
    
}
