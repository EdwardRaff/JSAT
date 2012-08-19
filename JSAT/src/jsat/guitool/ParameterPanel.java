package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.*;
import java.util.List;
import javax.swing.GroupLayout.Alignment;
import javax.swing.*;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import jsat.linear.distancemetrics.*;
import jsat.parameters.*;

/**
 * Parameter Panel provides a default GUI to alter the parameter values of any 
 * object that implements {@link Parameterized}. For {@link MetricParameter}, it
 * provides a sub list of possible parameters to choose from.
 * 
 * @author Edward Raff
 */
public class ParameterPanel extends javax.swing.JPanel
{
    final GridLayout gridLayout;
    final DistanceMetric[] distanceMetrics = new DistanceMetric[]
    {
        new EuclideanDistance(), 
        new ManhattanDistance(),
        new ChebyshevDistance(),
        new CosineDistance(),
        new MahalanobisDistance(),
    };
    
    /**
     * Creates new form ParameterPanel
     */
    public ParameterPanel(Parameterized parameterized)
    {
        initComponents();
        List<Parameter> parameters = parameterized.getParameters();
        gridLayout = new GridLayout(parameters.size(), 1);
        jPanelParameters.setLayout(gridLayout);
        
        for(Parameter param : parameters)
        {
            final JPanel subPanel = new JPanel(new BorderLayout());
            subPanel.setBorder(BorderFactory.createTitledBorder(param.getName()));
            if(param instanceof IntParameter)
            {
                final IntParameter intParam = (IntParameter) param;
                
                final JCheckBox checkBox = new JCheckBox();
                checkBox.setSelected(true);
                checkBox.setEnabled(false);
                
                final JTextField textField = new JTextField(Integer.toString(intParam.getValue()));
                textField.getDocument().addDocumentListener(new DocumentListener() {

                    @Override
                    public void insertUpdate(DocumentEvent e)
                    {
                        change(e);
                    }

                    @Override
                    public void removeUpdate(DocumentEvent e)
                    {
                        change(e);
                    }

                    @Override
                    public void changedUpdate(DocumentEvent e)
                    {
                        change(e);
                    }
                    
                    public void change(DocumentEvent e)
                    {
                        try
                        {
                            int newVal = Integer.parseInt(textField.getText().trim());
                            boolean good = intParam.setValue(newVal);
                            if(good)
                            {
                                checkBox.setSelected(true);
                            }
                            else
                                badInput();
                        }
                        catch(Exception ex)
                        {
                            badInput();
                        }
                    }
                    
                    private void badInput()
                    {
                        checkBox.setSelected(false);
                    }
                });
                
                subPanel.add(textField, BorderLayout.CENTER);
                subPanel.add(checkBox, BorderLayout.EAST);
                jPanelParameters.add(subPanel);
            }
            else if(param instanceof DoubleParameter)
            {
                final DoubleParameter doubleParam = (DoubleParameter) param;
                
                final JCheckBox checkBox = new JCheckBox();
                checkBox.setSelected(true);
                checkBox.setEnabled(false);
                
                final JTextField textField = new JTextField(Double.toString(doubleParam.getValue()));
                textField.getDocument().addDocumentListener(new DocumentListener() {

                    @Override
                    public void insertUpdate(DocumentEvent e)
                    {
                        change(e);
                    }

                    @Override
                    public void removeUpdate(DocumentEvent e)
                    {
                        change(e);
                    }

                    @Override
                    public void changedUpdate(DocumentEvent e)
                    {
                        change(e);
                    }
                    
                    public void change(DocumentEvent e)
                    {
                        try
                        {
                            double newVal = Double.parseDouble(textField.getText().trim());
                            boolean good = doubleParam.setValue(newVal);
                            if(good)
                            {
                                checkBox.setSelected(true);
                            }
                            else
                                badInput();
                        }
                        catch(Exception ex)
                        {
                            badInput();
                        }
                    }
                    
                    private void badInput()
                    {
                        checkBox.setSelected(false);
                    }
                });
                
                subPanel.add(textField, BorderLayout.CENTER);
                subPanel.add(checkBox, BorderLayout.EAST);
                jPanelParameters.add(subPanel);
            }
            else if(param instanceof BooleanParameter)
            {
                final BooleanParameter boolParam = (BooleanParameter) param;
                
                final JCheckBox checkBox = new JCheckBox();
                checkBox.setSelected(boolParam.getValue());
                checkBox.setEnabled(true);
                checkBox.addItemListener(new ItemListener() {

                    @Override
                    public void itemStateChanged(ItemEvent e)
                    {
                        boolParam.setValue(checkBox.isSelected());
                    }
                });
                
                
                subPanel.add(checkBox, BorderLayout.CENTER);
                jPanelParameters.add(subPanel);
            }
            else if(param instanceof ObjectParameter)
            {
                final ObjectParameter objParam = (ObjectParameter) param;
                
                final JComboBox comboBox = new JComboBox(objParam.parameterOptions().toArray());
                comboBox.setSelectedItem(objParam.getObject());
                comboBox.addActionListener(new ActionListener()
                {

                    @Override
                    public void actionPerformed(ActionEvent e)
                    {
                        objParam.setObject(comboBox.getSelectedItem());
                    }
                });
                subPanel.add(comboBox, BorderLayout.CENTER);
                jPanelParameters.add(subPanel);
            }
            else if(param instanceof MetricParameter)
            {
                final MetricParameter metricParam = (MetricParameter) param;
                
                final JComboBox comboBox = new JComboBox(distanceMetrics);
                for(int i = 0; i < distanceMetrics.length; i++)
                    if(distanceMetrics[i].toString().equals(metricParam.getMetric().toString()))
                        comboBox.setSelectedIndex(i);
                comboBox.addActionListener(new ActionListener()
                {

                    @Override
                    public void actionPerformed(ActionEvent e)
                    {
                        metricParam.setMetric(distanceMetrics[comboBox.getSelectedIndex()].clone());
                    }
                });
                subPanel.add(comboBox, BorderLayout.CENTER);
                jPanelParameters.add(subPanel);
            }
        }
    }

    /**
     * Obtains the jButton for the OK button, so that custom behavior may be 
     * added, or the button hidden if desired. 
     * 
     * @return the "OK" button. 
     */
    public JButton getjButtonOk()
    {
        return jButtonOk;
    }
    
    

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jButtonOk = new JButton();
        jPanelParameters = new JPanel();

        jButtonOk.setText("Ok");

        jPanelParameters.setBorder(BorderFactory.createTitledBorder("Parameters"));

        GroupLayout jPanelParametersLayout = new GroupLayout(jPanelParameters);
        jPanelParameters.setLayout(jPanelParametersLayout);
        jPanelParametersLayout.setHorizontalGroup(
            jPanelParametersLayout.createParallelGroup(Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        jPanelParametersLayout.setVerticalGroup(
            jPanelParametersLayout.createParallelGroup(Alignment.LEADING)
            .addGap(0, 225, Short.MAX_VALUE)
        );

        GroupLayout layout = new GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(0, 313, Short.MAX_VALUE)
                        .addComponent(jButtonOk))
                    .addComponent(jPanelParameters, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(Alignment.LEADING)
            .addGroup(Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanelParameters, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(ComponentPlacement.RELATED)
                .addComponent(jButtonOk)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private JButton jButtonOk;
    private JPanel jPanelParameters;
    // End of variables declaration//GEN-END:variables
}
