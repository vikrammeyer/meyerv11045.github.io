---
title: "Machine Learning Overview"
permalink: /machinelearning/
lesson: 0
---
The following articles are intended to introduce the field of machine learning to beginners by explaining the math behind the algorithms in order to give a better intuition for how the algorithms work. The necessary prerequisite knowledge is multivariable calculus and beginner linear algebra (matrices, transposes, and matrix multiplication). These articles are in part from my summer internship at DLG where I collaborated with Huy T., Nam H., and Nguyen V. 

{% for lesson in site.machinelearning %}
  <h2>
    <a href="{{ lesson.url }}">{{ lesson.title }}</a>
  </h2>
{% endfor %}