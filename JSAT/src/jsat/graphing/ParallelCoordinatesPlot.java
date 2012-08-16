package jsat.graphing;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.Vec;

/**
 * A plot for visualizing multi dimensional data sets. 
 * 
 * @author Edward Raff
 */
public class ParallelCoordinatesPlot extends Graph2D
{
    private DataSet dataSet;
    private double[] scales;
    private double[] offsets;
    /**
     * Stores the color for each category, or BLUE for no categories
     */
    private List<Color> categoryColors;

    /**
     * Creates a new parallel plot
     * @param dataSet 
     */
    public ParallelCoordinatesPlot(DataSet dataSet)
    {
        super(0, 1, 0, 1);
        this.dataSet = dataSet;
        this.setDrawMarkers(false);
        scales = new double[dataSet.getNumNumericalVars()];
        offsets = new double[scales.length];
        
        for(int i = 0; i < dataSet.getNumNumericalVars(); i++)
        {
            Vec v = dataSet.getNumericColumn(i);
            offsets[i] = v.min();
            scales[i] = v.max()-offsets[i];
        }
        
        //Set colors
        categoryColors = new ArrayList<Color>();
        if(dataSet instanceof ClassificationDataSet)
        {
            //TODO extract this out to some helper class
            ClassificationDataSet cds = (ClassificationDataSet) dataSet;
            float colorFactor = 1.0f/cds.getClassSize();
            for(int i = 0; i < cds.getClassSize(); i++)
            {
                Color c = Color.getHSBColor(i*colorFactor, 0.95f, 0.7f);
                categoryColors.add(c);
            }
        }
        else
            categoryColors.add(Color.BLUE);
    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        
        //Draw parallel lines
        double factor = 1.0/(dataSet.getNumNumericalVars()-1);
        g.setColor(Color.BLACK);
        double weigthRange = imageWidth-PAD*2;
        int[] xPixels = new int[dataSet.getNumNumericalVars()];
        for(int i = 0; i < dataSet.getNumNumericalVars(); i++)
        {
            xPixels[i] = (int) (PAD+weigthRange*(factor*i));
            g.drawLine(xPixels[i], toYCord(0, imageHeight), xPixels[i], toYCord(1, imageHeight));
        }
        
        //Draw Lines
        g.setColor(categoryColors.get(0));
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            if(dataSet instanceof ClassificationDataSet)
                g.setColor(categoryColors.get((((ClassificationDataSet) dataSet).getDataPointCategory(i))));
            Vec v = dataSet.getDataPoint(i).getNumericalValues();
            for(int j = 0; j < v.length()-1; j++)
            {
                g.drawLine(xPixels[j], 
                           toYCord( (v.get(j)-offsets[j])/scales[j], 
                                    imageHeight), 
                           xPixels[j+1], 
                           toYCord( (v.get(j+1)-offsets[j+1])/scales[j+1], 
                                    imageHeight));
            }
        }
        
        //Draw Labels
        g.setColor(Color.BLACK);
        FontMetrics fm = g.getFontMetrics();
        final int y = fm.getHeight();
        for(int i = 0; i < dataSet.getNumNumericalVars(); i++)
        {
            String catName = dataSet.getNumericName(i);
            int x = xPixels[i]-fm.stringWidth(catName)/2;
            g.drawString(catName, x, y);
        }
    }
    
}
