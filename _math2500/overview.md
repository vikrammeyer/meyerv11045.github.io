---
title: "Math 2500 Overview"
permalink: /math2500/
lesson: 0
---
Welcome to the beginning of my class notes for Math 2500 Fall 2020 at Vanderbilt University taught by Professor Bruce Hughes. This course covers Multivariable Calculus and Linear Algebra from a proofs based perspective.

{% for lesson in site.math2500 %}
  <h2>
    <a href="{{ lesson.url }}">{{ lesson.title }}</a>
  </h2>
{% endfor %}