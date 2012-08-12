
package jsat.graphing;

import java.awt.*;
import javax.swing.*;
import jsat.DataSet;
import jsat.linear.Vec;

/**
 * A matrix of scatter plots showing the relations between each combination of attributes in the data set
 * @author Edward Raff
 */
public class ScatterplotMatrix extends JDialog
{

    /**
     * Creates a new ScatterplotMatrix
     * @param parent the parent frame
     * @param title the title to use for the dialog
     * @param dataSet the data set to create plots for
     */
    public ScatterplotMatrix(Frame parent, String title, DataSet dataSet)
    {
        super(parent, title, false);

        JPanel panel = new JPanel();
        int numerVals = dataSet.getNumNumericalVars();
        panel.setLayout(new GridLayout(numerVals, numerVals));

        for(int i = 0; i < numerVals; i++)
        {
            Vec yAxis = dataSet.getNumericColumn(i);
            for(int j = 0; j < numerVals; j++)
            {
                if(i == j)
                {
                    
                    JLabel tmp = new JLabel(dataSet.getNumericName(i), JLabel.CENTER);
                    tmp.setBorder(BorderFactory.createLineBorder(Color.black));
                    panel.add(tmp);
                    continue;
                }

                Vec xAxis = dataSet.getNumericColumn(j);

                ScatterPlot sp = new ScatterPlot(xAxis, yAxis);
                sp.setPadding(0);
                sp.setBorder(BorderFactory.createLineBorder(Color.black));
                panel.add(sp);

            }
        }

        this.setContentPane(panel);
    }

}
